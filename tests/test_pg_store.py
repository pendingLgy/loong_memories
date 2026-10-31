'''pg_store.py 测试：PGVectorStore 与 AsyncPGVectorStore 的公共契约与业务方法。

覆盖策略：
1. 全方法覆盖：构造方法 __init__、backend 与 metric_type 类属性，以及 start、create_col、
   insert、search、keyword_search、delete、update、get、list 九个公共方法。
2. 作用域参数：写入靠 payload 内的 tenant_id / user_id；读取、检索、删除与更新以 tenant_id
   定位租户，user_id 可选；附加过滤统一用 filter 字典传入，与 pg_store.py 当前签名一致。
3. 向量随机：_vector 返回 TEST_VECTOR_SIZE 维的完全随机向量，每一维都独立取值。
   调用方的 seed 一律由 _random_seed() 从系统熵取随机整数，用例里不写任何固定数字 seed，
   因此跨用例、跨运行的向量方向都不同；同一用例内需要命中同一条记录时，
   把取到的 seed 存进局部变量复用，依靠同一 seed 逐位相同保证写入向量与查询向量一致。
4. 数据唯一：主键与 tenant_id / user_id 全部取自 _next_id()，它由系统熵基数加严格自增构成，
   保证同一次运行内绝不重复，跨次运行也不会落在同一号段。
5. 文本随机：payload 的 text 与 text_lemma 缺省时从 _TEXTS 的 100 条内容里随机取，
   这 100 条跨领域、跨长度、跨句式，两两之间几乎不共享字符二元组；
   需要区分前后值的场景用 _distinct_texts 无放回抽样，断言只比对写进去的那一条。
6. 过滤字典写法：search、keyword_search、list 会把 tenant_id 就地写入传入的 filter，
   而父类 _build_search_tenant_filter 遇到空字典会另建字典，导致租户条件丢失；
   因此统一用 _scope(tenant_id, ...) 传入已含 tenant_id 的非空字典。
   父类同处还会把 tenant_id 写进 user_id 维度，故这三个方法不传 user_id 形参，
   用户维度的隔离改由 filter 里的 user_id 键显式表达。
7. 继承契约验证：AsyncPGVectorStore 复用同步版的 SQL 构造与结果解析，其公共方法在异步
   子类实例上单独验证，不与同步类混用。
8. 不单独测试私有方法：_get_table、_col、_build_sql_*、_parse、_session 等不做直接断言，
   其逻辑由 insert、search、list、get 的行为间接验证；也不为 VectorStoreBase 建测试套件。

PG 向量库是外部服务，故整个模块以 integration 标记隔离：
未传 --integration 时整体跳过；异步用例另需 asyncpg 与 pgvector 扩展。
连接参数取自 conftest 的 vector 分区，可用 LOONG_TEST_VECTOR_PG_DSN 覆盖。

运行：
    python -m pytest tests --integration -q -k pg_store
'''

from __future__ import annotations

import itertools
import os
import random
import sys
from datetime import datetime, timezone

import pytest

try:
    import asyncpg
    import pytest_asyncio
    import sqlalchemy

    from conftest import CONFIG_VECTOR_SETTINGS
    from loong_memories.core.exceptions import OptionalDependencyError, StorageError
    from loong_memories.storage.memories.base import VectorStoreBase
    from loong_memories.storage.memories.pg_store import AsyncPGVectorStore, PGVectorStore
    from loong_memories.storage.memories.registry_vector import (
        is_vector_store_registered,
        resolve_vector_store,
    )

except ImportError as _exc:
    pytest.skip(f'[SKIP INFO] File {__file__}， 缺少 {str(_exc)}', allow_module_level=True)


pytestmark = pytest.mark.integration


if '--integration' not in sys.argv:
    pytest.skip(f'[SKIP INFO] File {__file__}：integration 集成测试，默认跳过（请使用 --integration 参数运行）',
                allow_module_level=True)


# ---------- 常量与辅助 ----------

# 集成用例专用集合名与维度：PG 中集合即表，create_col 里 metadata 的 GIN 索引固定写在
# vut_vectors 上，故集合名必须保持该值；维度取小值以减少写入数据量。
TEST_COLLECTION = 'vut_vectors'
TEST_VECTOR_SIZE = 8

# 进程内唯一的正整数生成器：基数取系统熵，之后严格自增。
# 基数每次运行都不同，避免撞上历史数据；自增保证同一次运行内绝不重复。
_ID_BASE = random.SystemRandom().randrange(1, 2 ** 40)
_ID_SEQ = itertools.count(1)

# 系统熵随机源：向量 seed 也走系统熵，用例里不出现固定数字 seed。
_TRNG = random.SystemRandom()


def _next_id():
    '''返回本次运行内唯一的正整数：主键与 tenant_id / user_id 共用同一发号器。'''
    return _ID_BASE + next(_ID_SEQ)


def _random_seed():
    '''返回一次性的完全随机 seed：每次调用都从系统熵取值。'''
    return _TRNG.randrange(1, 2 ** 62)


def _vector(seed):
    '''构造 TEST_VECTOR_SIZE 维的完全随机向量：每一维独立取值，同一个 seed 逐位相同。

    seed 一律来自 _random_seed()，因此不同调用、不同运行之间不会重合；
    需要查询命中同一条记录时，复用同一个 seed 才能保证向量逐位相同。
    '''
    rnd = random.Random(seed)
    return [rnd.uniform(-1.0, 1.0) for _ in range(TEST_VECTOR_SIZE)]

# 文本池：100 条跨领域、跨长度、跨句式的异构文本，_payload 不指定 text 时随机取一条。
# 刻意不写成同一类句子：每条来自不同主题（天文、地理、化学、生物、音乐、材质、
# 日常动作、英文、疑问句、命令式等），长度从七字到四十余字，字面几乎不共享关键词。
_TEXTS = [
'银河系中心存在一个超大质量黑洞，质量约为太阳的四百万倍',
'马里亚纳海沟最深处约有一万一千米',
'水的沸点在标准大气压下是一百摄氏度',
'贝多芬第九交响曲的第四乐章加入了合唱',
'番茄在植物学上属于浆果',
'马拉松全程为四十二点一九五公里',
'光在真空中的速度约为每秒三十万公里',
'大熊猫以竹子为主食，每天进食十小时以上',
'第一次世界大战始于一九一四年',
'蜂鸟是唯一能悬停飞行的鸟类',
'请把冰箱温度调到四度',
'这段代码在第三行少了一个分号',
'雨水顺着屋檐流进了水桶',
'他决定把旧自行车改装成电动款',
'舞台灯光突然熄灭了三秒',
'面粉与水的比例大约是三比一',
'地铁末班车在二十三点零七分发车',
'这本小说用倒叙手法讲述了三十年跨度',
'无人机在麦田上方盘旋了十五分钟',
'火山灰导致附近机场关闭两天',
'The quick brown fox jumps over the lazy dog',
'Ctrl Alt Delete 三个键同时按下可重启图形界面',
'葡萄糖的化学式是 C6H12O6',
'每年三月樱花从南向北依次开放',
'珊瑚白化与海水升温直接相关',
'围棋棋盘上有三百六十一个交叉点',
'他把咖啡洒在了会议记录上',
'台风眼附近的风力反而最小',
'圆珠笔的墨水靠重力与毛细作用流出',
'秦岭是南北气候的重要分界线',
'凌晨四点的面包房已经开始揉面',
'光纤依靠全反射传递信号',
'这只猫每天早上六点准时踩醒主人',
'银杏被称为植物界的活化石',
'用螺丝刀拧下背板四角的螺钉',
'沙漠昼夜温差可以超过三十摄氏度',
'管风琴的音管长度决定音高',
'快递包裹在转运中心滞留了两天',
'石墨与金刚石都由碳原子构成',
'潮汐主要由月球引力引起',
'他在黑板上写下了三个未知数',
'雪线高度随纬度升高而降低',
'老式收音机需要拉出天线才有信号',
'青霉素的发现源于一次培养皿污染',
'早高峰的换乘通道挤得走不动',
'蜜蜂通过摆尾舞传递花源方位',
'把烤箱预热到一百八十度再放入面糊',
'赤道附近的信风常年自东向西吹',
'他用手电筒照了照床底',
'石英钟的走时依赖晶体振荡频率',
'北极燕鸥每年往返两极之间',
'这段旋律用了三拍子的节奏',
'厨房下水道被油污堵住了',
'心电图记录的是心肌的电活动',
'高速公路上每隔两公里设有紧急电话',
'露水在草叶上凝成细小的珠子',
'他用橡皮擦掉了整段算式',
'地球自转一周约需二十三小时五十六分',
'铸铁锅需要先养出油膜才不粘',
'蚂蚁能举起自身体重数十倍的重物',
'图书馆的还书箱在门口左侧',
'臭氧层位于平流层的中上部',
'她把这幅水彩画挂在走廊尽头',
'体温计里的水银柱随温度上升',
'城市热岛效应让夜间降温变慢',
'小提琴的琴弓用马尾毛制成',
'这笔订单需要客户签字确认',
'珊瑚礁只占海洋面积的不到百分之一',
'他把闹钟提前了二十分钟',
'干冰升华时直接变成气体',
'月球表面的环形山大多来自撞击',
'空调滤网每月应当清洗一次',
'长颈鹿的颈椎数量与人类相同',
'钢琴调律需要把每个音对准基准频率',
'雨刷器在挡风玻璃上划出扇形',
'深海热泉附近存在化能合成生态',
'他把钥匙落在了办公室抽屉里',
'蜂蜜在低温下会自然结晶',
'高原上水的沸点明显低于一百度',
'这条隧道的通风机组装在天花板',
'蝙蝠靠回声定位在夜间捕食',
'她把最后一块拼图放进了缺口',
'混凝土需要养护七天才能达到强度',
'沙漏的流速由孔径与砂粒决定',
'火山岩里能看到气孔构造',
'他按下暂停键又倒回了十秒',
'洋流把暖水带向高纬度海区',
'毛笔的笔锋分狼毫与羊毫',
'这台打印机的硒鼓该换了',
'彩虹是阳光在水滴中折射与反射',
'他顺手把窗台上的花盆转了半圈',
'锂电池在低温下容量会明显下降',
'陨石坑周围的岩石常见冲击变质',
'她数了三遍还是少一张发票',
'松脂包裹昆虫可以形成琥珀',
'冰箱门封条老化会导致结霜',
'雪花多数是六角对称的枝状晶体',
'他把自行车链条重新上了一层油',
'地下水位下降会造成地面沉降',
'蚂蚁窝的入口常常堆着细沙粒',
]


def _random_text():
    '''从文本池随机取一条。'''
    return random.choice(_TEXTS)


def _distinct_texts(count):
    '''无放回抽取 count 条互不相同的文本，供需要区分前后值的用例使用。'''
    return random.sample(_TEXTS, count)


def _payload(tenant_id, user_id, text=None, text_lemma=None, **extra):
    '''组装一条写入用的 payload：作用域列、元数据 JSONB 与文本字段齐备。'''
    now = datetime.now(timezone.utc)
    content = _random_text() if text is None else text
    payload = dict(
        tenant_id=tenant_id,
        user_id=user_id,
        tag=content,
        text=content,
        text_lemma=text_lemma or content,
        created_at_tz=now,
        updated_at_tz=now,
    )
    payload.update(extra)
    return payload


def _scope(tenant_id, **extra):
    '''构造 search、keyword_search、list 用的 filter：预置 tenant_id 保证字典非空。'''
    return dict(tenant_id=tenant_id, **extra)


# ---------- fixtures ----------


@pytest.fixture(scope='module')
def vector_settings():
    '''vector 分区配置：取自 conftest 常量，逐模块深拷贝避免互相污染。'''
    return CONFIG_VECTOR_SETTINGS.model_copy(deep=True)


# ---------- 契约层：PGVectorStore ----------


class TestPGVectorStore:
    '''PGVectorStore 的公共契约：backend 登记、构造参数、metric_type 与缺依赖报错。'''

    def test_backend_is_registered(self):
        assert PGVectorStore.backend == 'pg'
        assert is_vector_store_registered('pg') is True
        assert resolve_vector_store('pg') is PGVectorStore

    def test_inherits_store_contract(self):
        assert issubclass(PGVectorStore, VectorStoreBase)
        assert isinstance(PGVectorStore(), VectorStoreBase)

    def test_init_defaults(self):
        store = PGVectorStore()
        assert store.connection_string is None
        assert (store.collection_name, store.vector_size) == ('', 1024)
        assert (store.database, store.user, store.password) == ('postgres', 'postgres', '')
        assert (store.host, store.port) == ('localhost', 5432)
        assert (store.use_hnsw, store.use_diskann, store.print_sql) == (True, False, False)
        assert (store.pool_size, store.max_overflow, store.pool_timeout) == (10, 20, 30)
        assert (store.pool_recycle, store.pool_pre_ping) == (1800, True)
        assert store.sslmode == 'prefer'
        assert store.connect_timeout == 5
        assert (store.application_name, store.options) == ('', '')
        assert store.target_session_attrs == 'any'

    def test_init_stores_conftest_values(self, vector_settings):
        pg, pool = vector_settings.pg, vector_settings.pool_config
        store = PGVectorStore(
            connection_string=pg.connection_string,
            collection_name=vector_settings.collection,
            vector_size=vector_settings.vector_size,
            database=pg.database,
            user=pg.user,
            password=pg.password,
            host=pg.host,
            port=pg.port,
            use_hnsw=pg.use_hnsw,
            use_diskann=pg.use_diskann,
            print_sql=pool.print_sql,
            pool_size=pool.pool_size,
            max_overflow=pool.max_overflow,
            pool_timeout=pool.pool_timeout,
            pool_recycle=pool.pool_recycle,
            pool_pre_ping=pool.pool_pre_ping,
            sslmode=pg.sslmode,
            connect_timeout=pg.connect_timeout,
            application_name=pg.application_name,
            options=pg.options,
            target_session_attrs=pg.target_session_attrs,
        )
        assert store.collection_name == vector_settings.collection
        assert store.vector_size == vector_settings.vector_size
        assert (store.database, store.user, store.password) == (pg.database, pg.user, pg.password)
        assert (store.host, store.port) == (pg.host, pg.port)
        assert (store.use_hnsw, store.use_diskann) == (pg.use_hnsw, pg.use_diskann)
        assert store.sslmode == pg.sslmode
        assert store.connect_timeout == pg.connect_timeout

    def test_init_builds_empty_metadata(self):
        assert list(PGVectorStore().metadata_obj.tables) == []

    def test_metric_type_is_cosine(self):
        assert PGVectorStore.metric_type == 'COSINE'

    def test_start_without_sqlalchemy_raises_optional_dependency_error(self, monkeypatch):
        monkeypatch.setattr('loong_memories.storage.memories.pg_store._SQLALCHEMY_AVAILABLE', False)
        with pytest.raises(OptionalDependencyError):
            PGVectorStore().start()

    def test_illegal_collection_name_raises_storage_error(self):
        '''集合名校验由 _col 把关，经 insert 入口间接触发，非法名在拼 SQL 前即被拦下。'''
        store = PGVectorStore(collection_name='bad name!')
        with pytest.raises(StorageError):
            store.insert([_vector(_random_seed())], [_payload(_next_id(), _next_id())], [_next_id()])


class TestAsyncPGVectorStore:
    '''AsyncPGVectorStore 的公共契约：backend 登记、构造参数与缺依赖报错。'''

    def test_backend_is_registered(self):
        assert AsyncPGVectorStore.backend == 'pg_async'
        assert is_vector_store_registered('pg_async') is True
        assert resolve_vector_store('pg_async') is AsyncPGVectorStore

    def test_inherits_sync_store(self):
        assert issubclass(AsyncPGVectorStore, PGVectorStore)
        assert issubclass(AsyncPGVectorStore, VectorStoreBase)

    def test_init_defaults(self):
        store = AsyncPGVectorStore()
        assert store.connection_string is None
        assert (store.collection_name, store.vector_size) == ('', 1024)
        assert (store.host, store.port) == ('localhost', 5432)
        assert (store.use_hnsw, store.use_diskann, store.print_sql) == (True, False, False)
        assert (store.pool_size, store.max_overflow, store.pool_timeout) == (10, 20, 30)
        assert store.sslmode == 'prefer' and store.connect_timeout == 5

    def test_init_stores_conftest_values(self, vector_settings):
        pg = vector_settings.pg
        store = AsyncPGVectorStore(
            connection_string=pg.connection_string,
            collection_name=TEST_COLLECTION,
            vector_size=TEST_VECTOR_SIZE,
            database=pg.database,
            user=pg.user,
            password=pg.password,
            host=pg.host,
            port=pg.port,
            sslmode=pg.sslmode,
            target_session_attrs=pg.target_session_attrs,
        )
        assert store.collection_name == TEST_COLLECTION
        assert store.vector_size == TEST_VECTOR_SIZE
        assert store.target_session_attrs == pg.target_session_attrs

    def test_metric_type_is_cosine(self):
        assert AsyncPGVectorStore.metric_type == 'COSINE'

    async def test_start_without_sqlalchemy_raises_optional_dependency_error(self, monkeypatch):
        monkeypatch.setattr('loong_memories.storage.memories.pg_store._SQLALCHEMY_AVAILABLE', False)
        with pytest.raises(OptionalDependencyError):
            await AsyncPGVectorStore().start()


# ---------- 真实服务：PGVectorStore ----------


class TestPGVectorStoreIntegration:
    '''PGVectorStore 在真实 PostgreSQL 上的公共方法行为：建表、写入、读取、检索、更新与删除。

    连接参数取自 conftest 的 vector 分区，可用 LOONG_TEST_VECTOR_PG_DSN 覆盖；集合名固定为
    TEST_COLLECTION（表名），每个用例的 tenant_id / user_id 与主键均由 _next_id() 唯一发放，
    因此并发与重复运行都不会互相踩到同一条记录。'''

    @pytest.fixture(scope='module')
    def pg_store(self, vector_settings):
        '''建连并确保表、向量索引与全文索引存在（start 内部会调用 create_col）。'''
        pg, pool = vector_settings.pg, vector_settings.pool_config
        dsn = os.environ.get('LOONG_TEST_VECTOR_PG_DSN', '') or pg.connection_string
        store = PGVectorStore(
            connection_string=dsn or None,
            collection_name=TEST_COLLECTION,
            vector_size=TEST_VECTOR_SIZE,
            database=pg.database,
            user=pg.user,
            password=pg.password,
            host=pg.host,
            port=pg.port,
            use_hnsw=pg.use_hnsw,
            use_diskann=pg.use_diskann,
            print_sql=pool.print_sql,
            pool_size=pool.pool_size,
            max_overflow=pool.max_overflow,
            pool_timeout=pool.pool_timeout,
            pool_recycle=pool.pool_recycle,
            pool_pre_ping=pool.pool_pre_ping,
            sslmode=pg.sslmode,
            connect_timeout=pg.connect_timeout,
        )
        try:
            store.start()
        except Exception as exc:
            pytest.skip('无法连接 conftest 配置的 PostgreSQL：' + str(exc))
        yield store

        store.engine.dispose()

    def test_start_returns_self(self, pg_store):
        assert isinstance(pg_store, PGVectorStore)
        assert pg_store.engine is not None

    def test_create_col_is_idempotent(self, pg_store):
        '''表已存在时 create_col 直接返回，不重建索引也不报错。'''
        assert pg_store.create_col(TEST_COLLECTION, TEST_VECTOR_SIZE) is None

    def test_insert_then_get_returns_record(self, pg_store):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        pg_store.insert([_vector(_random_seed())], [_payload(tenant_id, user_id)], [rid])
        got = pg_store.get(rid, tenant_id, user_id)
        assert got is not None
        assert (got.id, got.tenant_id, got.user_id) == (rid, tenant_id, user_id)
        assert got.text in _TEXTS

    def test_get_unknown_id_returns_none(self, pg_store):
        assert pg_store.get(_next_id(), _next_id()) is None

    def test_get_with_other_user_returns_none(self, pg_store):
        '''user_id 参与定位：换一个用户读取同一 id 应当查不到。'''
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        pg_store.insert([_vector(_random_seed())], [_payload(tenant_id, user_id)], [rid])
        assert pg_store.get(rid, tenant_id, _next_id()) is None

    def test_insert_persists_text_lemma(self, pg_store):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        pg_store.insert([_vector(_random_seed())],
                        [_payload(tenant_id, user_id, text_lemma='milvus vector database')], [rid])
        got = pg_store.get(rid, tenant_id, user_id)
        assert got.text in _TEXTS and got.text_lemma == 'milvus vector database'

    def test_created_at_is_serialized_to_iso_string(self, pg_store):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        pg_store.insert([_vector(_random_seed())], [_payload(tenant_id, user_id)], [rid])
        got = pg_store.get(rid, tenant_id, user_id)
        assert isinstance(got.created_at_tz, str)
        assert isinstance(got.updated_at_tz, str)

    def test_insert_duplicate_id_raises_integrity_error(self, pg_store):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        pg_store.insert([_vector(_random_seed())], [_payload(tenant_id, user_id)], [rid])
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            pg_store.insert([_vector(_random_seed())], [_payload(tenant_id, user_id)], [rid])

    def test_insert_dimension_mismatch_raises_error(self, pg_store):
        '''向量维度与列定义不一致时由数据库拒绝，异常原样抛给调用方。'''
        bad_vector = _vector(_random_seed())[:-1]
        with pytest.raises(sqlalchemy.exc.SQLAlchemyError):
            pg_store.insert([bad_vector], [_payload(_next_id(), _next_id())], [_next_id()])

    def test_search_returns_hits_desc_by_score(self, pg_store):
        tenant_id, user_id = _next_id(), _next_id()
        ids = [_next_id() for _ in range(3)]
        seeds = [_random_seed() for _ in range(3)]
        pg_store.insert([_vector(seed) for seed in seeds],
                        [_payload(tenant_id, user_id) for _ in range(3)], ids)
        hits = pg_store.search(_random_text(), _vector(seeds[0]), tenant_id,
                               filter=_scope(tenant_id), top_k=3)
        assert len(hits) == 3
        assert sorted([hit.id for hit in hits]) == sorted(ids)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_top_k(self, pg_store):
        tenant_id, user_id = _next_id(), _next_id()
        seed = _random_seed()
        pg_store.insert([_vector(seed) for _ in range(3)],
                        [_payload(tenant_id, user_id) for _ in range(3)],
                        [_next_id() for _ in range(3)])
        hits = pg_store.search(_random_text(), _vector(seed), tenant_id,
                               filter=_scope(tenant_id), top_k=2)
        assert len(hits) == 2

    def test_search_filters_by_tenant(self, pg_store):
        '''tenant_id 由入参指定并写入过滤表达式，跨租户数据不得互相召回。'''
        text_a, text_b = _distinct_texts(2)
        first, second, user_id = _next_id(), _next_id(), _next_id()
        seed = _random_seed()
        pg_store.insert([_vector(seed), _vector(seed)],
                        [_payload(first, user_id, text_a), _payload(second, user_id, text_b)],
                        [_next_id(), _next_id()])
        hits = pg_store.search(text_a, _vector(seed), first, filter=_scope(first), top_k=10)
        assert [hit.text for hit in hits] == [text_a]

    def test_search_filters_on_metadata_field(self, pg_store):
        '''非白名单键落到 metadata JSONB 上做包含过滤。'''
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        pg_store.insert([_vector(_random_seed())], [_payload(tenant_id, user_id, tag='red')], [rid])
        hits = pg_store.search(_random_text(), _vector(_random_seed()), tenant_id,
                               filter=_scope(tenant_id, tag='red'), top_k=10)
        assert rid in [hit.id for hit in hits]

    def test_search_rejects_unsupported_operator(self, pg_store):
        '''非法操作符在拼装 WHERE 条件时就被拦下，不落到数据库。'''
        with pytest.raises(ValueError):
            pg_store.search('x', _vector(_random_seed()), _next_id(),
                           filter=_scope(_next_id(), id=dict(bad_op=1)))

    def test_search_filters_by_user_column(self, pg_store):
        '''user_id 是顶层列，可经 filter 显式限定用户维度。'''
        tenant_id, user_a, user_b = _next_id(), _next_id(), _next_id()
        rid = _next_id()
        pg_store.insert([_vector(_random_seed())], [_payload(tenant_id, user_a)], [rid])
        mine = pg_store.search(_random_text(), _vector(_random_seed()), tenant_id,
                               filter=_scope(tenant_id, user_id=user_a), top_k=10)
        assert rid in [hit.id for hit in mine]
        hits = pg_store.search(_random_text(), _vector(_random_seed()), tenant_id,
                               filter=_scope(tenant_id, user_id=user_b), top_k=10)
        assert hits == []

    def test_keyword_search_matches_text_lemma(self, pg_store):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        pg_store.insert([_vector(_random_seed())],
                        [_payload(tenant_id, user_id, text_lemma='milvus vector database')], [rid])
        hits = pg_store.keyword_search('milvus', tenant_id, filter=_scope(tenant_id), top_k=5)
        assert hits is not None
        assert [hit.id for hit in hits] == [rid]

    def test_keyword_search_returns_empty_on_no_match(self, pg_store):
        tenant_id = _next_id()
        assert pg_store.keyword_search('zz' + str(_next_id()), tenant_id, filter=_scope(tenant_id)) == []

    def test_keyword_search_returns_none_on_failure(self, pg_store):
        '''全文检索刻意吞掉执行异常：非法过滤值触发数据库报错时返回 None。'''
        tenant_id = _next_id()
        bad = _scope(tenant_id, id=dict(gt='abc'))
        assert pg_store.keyword_search('milvus', tenant_id, filter=bad) is None

    def test_update_vector_and_payload(self, pg_store):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        text_before, text_after = _distinct_texts(2)
        pg_store.insert([_vector(_random_seed())], [_payload(tenant_id, user_id, text_before)], [rid])
        pg_store.update(rid, tenant_id, user_id,
                        vector=_vector(_random_seed()),
                        payload=_payload(tenant_id, user_id, text_after))
        got = pg_store.get(rid, tenant_id, user_id)
        assert got is not None and got.text == text_after
        assert got.metadata.get('tag') == text_after

    def test_update_keeps_existing_vector_when_omitted(self, pg_store):
        '''只传 payload 时沿用库中已有向量：用原向量再检索，相似度应为 1.0。'''
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        seed = _random_seed()
        text_before, text_after = _distinct_texts(2)
        pg_store.insert([_vector(seed)], [_payload(tenant_id, user_id, text_before)], [rid])
        pg_store.update(rid, tenant_id, user_id, payload=_payload(tenant_id, user_id, text_after))
        hits = pg_store.search(text_after, _vector(seed), tenant_id, filter=_scope(tenant_id), top_k=1)
        assert hits and hits[0].id == rid
        assert hits[0].score == pytest.approx(1.0, abs=1e-6)

    def test_update_without_params_is_noop(self, pg_store):
        '''vector 与 payload 都不传时直接返回，不产生任何 UPDATE。'''
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        text_before = _random_text()
        pg_store.insert([_vector(_random_seed())], [_payload(tenant_id, user_id, text_before)], [rid])
        assert pg_store.update(rid, tenant_id, user_id) is None
        assert pg_store.get(rid, tenant_id, user_id).text == text_before

    def test_update_with_wrong_tenant_is_noop(self, pg_store):
        '''tenant_id 参与定位：换一个租户更新同一 id 不应生效。'''
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        text_before, text_after = _distinct_texts(2)
        pg_store.insert([_vector(_random_seed())], [_payload(tenant_id, user_id, text_before)], [rid])
        pg_store.update(rid, _next_id(), user_id, payload=_payload(tenant_id, user_id, text_after))
        assert pg_store.get(rid, tenant_id, user_id).text == text_before

    def test_delete_removes_record(self, pg_store):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        pg_store.insert([_vector(_random_seed())], [_payload(tenant_id, user_id)], [rid])
        assert pg_store.delete(rid, tenant_id, user_id) is None
        assert pg_store.get(rid, tenant_id, user_id) is None

    def test_delete_with_wrong_tenant_keeps_record(self, pg_store):
        '''tenant_id 参与定位：换一个租户删除同一 id 不应生效。'''
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        pg_store.insert([_vector(_random_seed())], [_payload(tenant_id, user_id)], [rid])
        pg_store.delete(rid, _next_id(), user_id)
        assert pg_store.get(rid, tenant_id, user_id) is not None

    def test_list_respects_top_k(self, pg_store):
        tenant_id, user_id = _next_id(), _next_id()
        seed = _random_seed()
        pg_store.insert([_vector(seed) for _ in range(3)],
                        [_payload(tenant_id, user_id) for _ in range(3)],
                        [_next_id() for _ in range(3)])
        assert len(pg_store.list(tenant_id, filter=_scope(tenant_id), top_k=2)) == 2

    def test_list_applies_in_operator(self, pg_store):
        '''顶层列支持 in 操作符：只列出指定用户的数据。'''
        tenant_id, user_a, user_b = _next_id(), _next_id(), _next_id()
        text_a, text_b = _distinct_texts(2)
        pg_store.insert([_vector(_random_seed()), _vector(_random_seed())],
                        [_payload(tenant_id, user_a, text_a), _payload(tenant_id, user_b, text_b)],
                        [_next_id(), _next_id()])
        listed = pg_store.list(tenant_id, filter=_scope(tenant_id, user_id=dict([('in', [user_a])])))
        assert [item.text for item in listed] == [text_a]

    def test_list_filters_on_metadata_field(self, pg_store):
        '''非白名单键落到 metadata JSONB 上做包含过滤。'''
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        pg_store.insert([_vector(_random_seed())], [_payload(tenant_id, user_id, tag='blue')], [rid])
        listed = pg_store.list(tenant_id, filter=_scope(tenant_id, tag='blue'))
        assert rid in [item.id for item in listed]


# ---------- 真实服务：AsyncPGVectorStore ----------


@pytest.mark.asyncio(loop_scope='module')
class TestAsyncPGVectorStoreIntegration:
    '''AsyncPGVectorStore 在真实 PostgreSQL 上的公共方法行为，语义与同步版本一致。'''

    @pytest_asyncio.fixture(scope='module', loop_scope='module')
    async def async_pg_store(self, vector_settings):
        '''建连并确保表与索引存在（start 内部会 await create_col）。'''
        pg, pool = vector_settings.pg, vector_settings.pool_config
        dsn = os.environ.get('LOONG_TEST_VECTOR_PG_DSN', '') or pg.connection_string
        store = AsyncPGVectorStore(
            connection_string=dsn or None,
            collection_name=TEST_COLLECTION,
            vector_size=TEST_VECTOR_SIZE,
            database=pg.database,
            user=pg.user,
            password=pg.password,
            host=pg.host,
            port=pg.port,
            use_hnsw=pg.use_hnsw,
            use_diskann=pg.use_diskann,
            print_sql=pool.print_sql,
            pool_size=pool.pool_size,
            max_overflow=pool.max_overflow,
            pool_timeout=pool.pool_timeout,
            pool_recycle=pool.pool_recycle,
            pool_pre_ping=pool.pool_pre_ping,
            sslmode=pg.sslmode,
            connect_timeout=pg.connect_timeout,
        )
        try:
            await store.start()
        except Exception as exc:
            pytest.skip('无法连接 conftest 配置的 PostgreSQL：' + str(exc))
        yield store

        await store.engine.dispose()

    async def test_start_returns_self(self, async_pg_store):
        assert isinstance(async_pg_store, AsyncPGVectorStore)
        assert async_pg_store.engine is not None

    async def test_create_col_is_idempotent(self, async_pg_store):
        assert await async_pg_store.create_col(TEST_COLLECTION, TEST_VECTOR_SIZE) is None

    async def test_insert_then_get_returns_record(self, async_pg_store):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        await async_pg_store.insert([_vector(_random_seed())], [_payload(tenant_id, user_id)], [rid])
        got = await async_pg_store.get(rid, tenant_id, user_id)
        assert got is not None
        assert (got.id, got.tenant_id, got.user_id) == (rid, tenant_id, user_id)
        assert got.text in _TEXTS

    async def test_get_unknown_id_returns_none(self, async_pg_store):
        assert await async_pg_store.get(_next_id(), _next_id()) is None

    async def test_insert_persists_text_lemma(self, async_pg_store):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        await async_pg_store.insert([_vector(_random_seed())],
                                    [_payload(tenant_id, user_id, text_lemma='milvus vector database')],
                                    [rid])
        got = await async_pg_store.get(rid, tenant_id, user_id)
        assert got.text in _TEXTS and got.text_lemma == 'milvus vector database'

    async def test_search_returns_hits_desc_by_score(self, async_pg_store):
        tenant_id, user_id = _next_id(), _next_id()
        ids = [_next_id() for _ in range(3)]
        seeds = [_random_seed() for _ in range(3)]
        await async_pg_store.insert([_vector(seed) for seed in seeds],
                                    [_payload(tenant_id, user_id) for _ in range(3)], ids)
        hits = await async_pg_store.search(_random_text(), _vector(seeds[0]), tenant_id,
                                          filter=_scope(tenant_id), top_k=3)
        assert len(hits) == 3
        assert sorted([hit.id for hit in hits]) == sorted(ids)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    async def test_search_respects_top_k(self, async_pg_store):
        tenant_id, user_id = _next_id(), _next_id()
        seed = _random_seed()
        await async_pg_store.insert([_vector(seed) for _ in range(3)],
                                    [_payload(tenant_id, user_id) for _ in range(3)],
                                    [_next_id() for _ in range(3)])
        hits = await async_pg_store.search(_random_text(), _vector(seed), tenant_id,
                                          filter=_scope(tenant_id), top_k=2)
        assert len(hits) == 2

    async def test_search_filters_by_tenant(self, async_pg_store):
        text_a, text_b = _distinct_texts(2)
        first, second, user_id = _next_id(), _next_id(), _next_id()
        seed = _random_seed()
        await async_pg_store.insert([_vector(seed), _vector(seed)],
                                    [_payload(first, user_id, text_a), _payload(second, user_id, text_b)],
                                    [_next_id(), _next_id()])
        hits = await async_pg_store.search(text_a, _vector(seed), first, filter=_scope(first), top_k=10)
        assert [hit.text for hit in hits] == [text_a]

    async def test_keyword_search_matches_text_lemma(self, async_pg_store):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        await async_pg_store.insert([_vector(_random_seed())],
                                    [_payload(tenant_id, user_id, text_lemma='milvus vector database')],
                                    [rid])
        hits = await async_pg_store.keyword_search('milvus', tenant_id,
                                                  filter=_scope(tenant_id), top_k=5)
        assert hits is not None
        assert [hit.id for hit in hits] == [rid]

    async def test_keyword_search_returns_none_on_failure(self, async_pg_store):
        tenant_id = _next_id()
        bad = _scope(tenant_id, id=dict(gt='abc'))
        assert await async_pg_store.keyword_search('milvus', tenant_id, filter=bad) is None

    async def test_update_vector_and_payload(self, async_pg_store):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        text_before, text_after = _distinct_texts(2)
        await async_pg_store.insert([_vector(_random_seed())],
                                    [_payload(tenant_id, user_id, text_before)], [rid])
        await async_pg_store.update(rid, tenant_id, user_id,
                                    vector=_vector(_random_seed()),
                                    payload=_payload(tenant_id, user_id, text_after))
        got = await async_pg_store.get(rid, tenant_id, user_id)
        assert got is not None and got.text == text_after

    async def test_update_without_params_is_noop(self, async_pg_store):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        text_before = _random_text()
        await async_pg_store.insert([_vector(_random_seed())],
                                    [_payload(tenant_id, user_id, text_before)], [rid])
        assert await async_pg_store.update(rid, tenant_id, user_id) is None
        got = await async_pg_store.get(rid, tenant_id, user_id)
        assert got.text == text_before

    async def test_list_respects_top_k(self, async_pg_store):
        tenant_id, user_id = _next_id(), _next_id()
        seed = _random_seed()
        await async_pg_store.insert([_vector(seed) for _ in range(3)],
                                    [_payload(tenant_id, user_id) for _ in range(3)],
                                    [_next_id() for _ in range(3)])
        assert len(await async_pg_store.list(tenant_id, filter=_scope(tenant_id), top_k=2)) == 2

    async def test_list_filters_on_metadata_field(self, async_pg_store):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        await async_pg_store.insert([_vector(_random_seed())],
                                    [_payload(tenant_id, user_id, tag='blue')], [rid])
        listed = await async_pg_store.list(tenant_id, filter=_scope(tenant_id, tag='blue'))
        assert rid in [item.id for item in listed]

    async def test_delete_removes_record(self, async_pg_store):
        '''异步 delete 按 id 与租户条件删除，删除后按 id 查询应返回 None。'''
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        await async_pg_store.insert([_vector(_random_seed())],
                                    [_payload(tenant_id, user_id)], [rid])
        await async_pg_store.delete(rid, tenant_id, user_id)
        assert await async_pg_store.get(rid, tenant_id, user_id) is None

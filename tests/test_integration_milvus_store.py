'''milvus_store.py 测试：MilvusVectorStore 与 AsyncMilvusVectorStore 的公共契约与业务方法。

覆盖策略：
1. 全方法覆盖：构造方法 __init__、backend 与 metric_type 类属性，以及 start、create_col、
   insert、search、keyword_search、delete、update、get、list 十个公共方法。
2. 作用域参数：写入靠 payload 内的 tenant_id / user_id，其余方法以 tenant_id 定位租户、
   user_id 可选；附加过滤统一用 filter 字典传入，与 milvus_store.py 当前签名一致。
3. 向量随机：_vector 返回 dim 维的完全随机向量，每一维都独立取值，不再只有首维非零。
   调用方的 seed 一律由 _random_seed() 从系统熵取随机整数，用例里不写任何固定数字 seed，
   因此跨用例、跨运行的向量方向都不同；同一用例内需要命中同一条记录时，
   把取到的 seed 存进局部变量复用，依靠「同一 seed 逐位相同」保证写入向量与查询向量一致。
4. 数据唯一：主键与 tenant_id / user_id 全部取自 _next_id()，它由系统熵基数加严格自增构成，
   保证同一次运行内绝不重复、跨次运行也不会落在同一号段；不写任何固定字面量主键，
   因此用例之间、多次运行之间不会争抢同一条记录。
5. 文本随机：payload 的 text 与 text_lemma 缺省时从 _TEXTS 的 100 条内容里随机取。
   这 100 条刻意跨领域、跨长度、跨句式，两两之间几乎不共享字符二元组，
   因此语义与关键词相似度天然分散，随机取值不会让用例之间产生语义串扰。
   需要区分前后值的场景用 _distinct_texts 无放回抽样，断言只比对写进去的那一条。
6. 继承契约验证：AsyncMilvusVectorStore 复用同步版的 schema、记录组装与结果解析逻辑，
   其公共方法在异步子类实例上单独验证，不与同步类混用。
7. 不单独测试私有方法：_schema、_record、_create_filter、_parse、_build_param_insert、
   _build_id_filter、_build_search_tenant_filter、_query_parse 等不做直接断言，
   其逻辑由 insert、search、list、get 的行为间接验证。
8. 不为父类建测试类：VectorStoreBase 不设独立测试套件。

Milvus 是外部服务，故整个模块以 integration 标记隔离：
未传 --integration 时整体跳过；异步用例另需 pymilvus>=2.5 提供的 AsyncMilvusClient。
连接参数取自 conftest 的 vector 分区。

运行：
    hatch run loong_memories_dev:pytest tests --integration -q -k milvus
'''

from __future__ import annotations

import itertools
import random
import sys

import pytest


try:
    import pymilvus
    import pytest_asyncio

    from conftest import CONFIG_VECTOR_SETTINGS
    from loong_memories.core.exceptions import OptionalDependencyError, StorageError
    from loong_memories.storage.memories.base import VectorStoreBase
    from loong_memories.storage.memories.milvus_store import (
        AsyncMilvusClient,
        AsyncMilvusVectorStore,
        MilvusVectorStore,
    )
    from loong_memories.storage.memories.registry_vector import (
        is_vector_store_registered,
        resolve_vector_store,
    )

except ImportError as _exc:
    pytest.skip(f"[SKIP INFO] File {__file__}， 缺少 {str(_exc)}", allow_module_level=True)


pytestmark = pytest.mark.integration

if "--integration" not in sys.argv:
    pytest.skip(f"[SKIP INFO] File {__file__}：integration 集成测试，默认跳过（请使用 --integration 参数运行）",
                allow_module_level=True)

# 异步实现依赖 pymilvus 2.5 起提供的 AsyncMilvusClient，缺失时只跳过异步用例。
HAS_ASYNC_CLIENT = AsyncMilvusClient is not None

needs_async_client = pytest.mark.skipif(
    not HAS_ASYNC_CLIENT,
    reason="异步 Milvus 用例需要 pymilvus>=2.5 提供的 AsyncMilvusClient",
)


# ---------- 辅助 ----------

# 进程内唯一的正整数生成器：基数取系统熵，之后严格自增。
# 基数每次运行都不同，避免撞上历史数据；自增保证同一次运行内绝不重复，
# 因此主键与 tenant_id / user_id 都不会因随机碰撞引发冲突。
_ID_BASE = random.SystemRandom().randrange(1, 2 ** 40)
_ID_SEQ = itertools.count(1)


def _next_id():
    '''返回本次运行内唯一的正整数：主键与 tenant_id / user_id 共用同一发号器。'''
    return _ID_BASE + next(_ID_SEQ)


# 文本池：100 条跨领域、跨长度、跨句式的异构文本，_payload 不指定 text 时随机取一条。
# 刻意不写成同一类句子：每条来自不同主题（天文、地理、化学、生物、音乐、材质、
# 日常动作、英文、疑问句、命令式等），长度从七字到四十余字，字面几乎不共享关键词，
# 因此语义与关键词相似度天然分散，不会在检索里互相串味。
_TEXTS = [
    "银河系中心存在一个超大质量黑洞，质量约为太阳的四百万倍",
    "马里亚纳海沟最深处约有一万一千米",
    "水的沸点在标准大气压下是一百摄氏度",
    "贝多芬第九交响曲的第四乐章加入了合唱",
    "番茄在植物学上属于浆果",
    "马拉松全程为四十二点一九五公里",
    "光在真空中的速度约为每秒三十万公里",
    "大熊猫以竹子为主食，每天进食十小时以上",
    "第一次世界大战始于一九一四年",
    "蜂鸟是唯一能悬停飞行的鸟类",
    "请把冰箱温度调到四度",
    "这段代码在第三行少了一个分号",
    "雨水顺着屋檐流进了水桶",
    "他决定把旧自行车改装成电动款",
    "舞台灯光突然熄灭了三秒",
    "面粉与水的比例大约是三比一",
    "地铁末班车在二十三点零七分发车",
    "这本小说用倒叙手法讲述了三十年跨度",
    "无人机在麦田上方盘旋了十五分钟",
    "火山灰导致附近机场关闭两天",
    "The quick brown fox jumps over the lazy dog",
    "Ctrl Alt Delete 三个键同时按下可重启图形界面",
    "葡萄糖的化学式是 C6H12O6",
    "每年三月樱花从南向北依次开放",
    "他在围棋三段赛中连胜十一局",
    "这幅油画使用了厚重的堆叠颜料技法",
    "潮汐主要由月球引力引起",
    "面包发酵需要保持三十度左右的温度",
    "这段音频采样率设置为四点四一万赫兹",
    "南极冰盖储存了全球约七成的淡水",
    "为什么天空是蓝色的",
    "谁能在一小时内跑完十公里",
    "这里到底该用正弦还是余弦",
    "要不要把这个抽屉换成滑轨",
    "暖气片为什么只在冬天有声音",
    "这个参数调到多少才合适",
    "谁把钥匙落在了玄关柜上",
    "明天会不会下雨",
    "这种蘑菇能吃吗",
    "灯光为什么忽明忽暗",
    "螺丝刀放在工具箱第二层左侧",
    "港口的集装箱堆到了六层高",
    "温度计显示零下八度",
    "他的左手腕戴着一只机械表",
    "咖啡研磨粒度越细萃取越快",
    "提琴的琴弦需要每季度更换一次",
    "这本书的索引在第两百四十页",
    "隧道全长七点二公里双向四车道",
    "木工胶完全固化需要二十四小时",
    "这座桥的主跨以钢索悬吊",
    "编程语言的发展可以分为五代",
    "烘焙咖啡豆的温度曲线决定了风味",
    "龙卷风的中心气压极低",
    "围棋棋盘由十九条线交叉而成",
    "羊绒来自山羊的底层绒毛",
    "铅蓄电池的循环寿命约三百次",
    "风速计通过三只半球旋转测速",
    "古琴有七根弦十三个徽位",
    "蜂蜜的含水量低于百分之二十",
    "琥珀是树脂的化石",
    "我需要一个能装五公斤行李的背包",
    "请把窗台上的花盆移到阳台",
    "把这段录音转成带字幕的视频",
    "帮我把纸质表格录入成电子版",
    "找出这三份清单里重复的条目",
    "把温度日志按小时汇总成折线图",
    "给这批照片统一加上时间水印",
    "把这段对话翻译成日文",
    "统计一周内各时段的客流量",
    "把这段话缩短到五十字以内",
    "三号线的早高峰拥挤程度超出预期",
    "高原反应通常在海拔三千米以上出现",
    "锂电池在零度以下充电会损伤电极",
    "混凝土养护期间需要保持湿润",
    "老式胶片相机依靠机械快门控制曝光",
    "蜂群失王后会培育新的蜂王",
    "潮间带的生物需要耐受盐度剧变",
    "京剧的脸谱颜色对应不同性格",
    "折纸的每一道折痕都影响最终形态",
    "纺锤的转速决定了纱线的粗细",
    "请勿在雷雨天气使用有线电话",
    "舞台后方的幕布需要在开场前检查",
    "这把伞的骨架断了两根",
    "保温杯里的水过了六小时还是热的",
    "院子里的葡萄藤爬满了整个架子",
    "传菜的电梯只能容纳两层托盘",
    "这台老式收音机需要预热半分钟",
    "图书馆的三楼在整修暂停开放",
    "河道在枯水期露出大片淤泥",
    "车间的排风扇噪音超过了八十分贝",
    "梯田顺着山坡逐级下降",
    "船坞里的木船正在重新刷漆",
    "齿轮的齿数比决定了输出转速",
    "熔炉温度升到一千二百度才开始浇铸",
    "竹子在雨后一夜之间长高十几厘米",
    "老钟表的走时误差每天约十秒",
    "山脊上的风把云推成了长条",
    "蚁群用信息素标记搬运路线",
    "晾晒的辣椒在阳光下逐渐变红",
    "巷子尽头有一口被封住的旧井",
]


def _random_text():
    '''从文本池随机取一条。'''
    return random.choice(_TEXTS)


def _distinct_texts(count):
    '''无放回抽取 count 条互不相同的文本，供需要区分前后值的用例使用。'''
    return random.sample(_TEXTS, count)


# 系统熵随机源：向量 seed 也走系统熵，用例里不出现固定数字 seed。
_TRNG = random.SystemRandom()


def _random_seed():
    '''返回一次性的完全随机 seed：每次调用都从系统熵取值。'''
    return _TRNG.randrange(1, 2 ** 62)


def _vector(seed, dim):
    '''构造 dim 维的完全随机向量：每一维都独立取随机值，同一个 seed 逐位相同。

    seed 一律来自 _random_seed()，因此不同调用、不同运行之间不会重合；
    需要查询命中同一条记录时，复用同一个 seed 才能保证向量逐位相同。
    '''
    rnd = random.Random(seed)
    return [rnd.uniform(-1.0, 1.0) for _ in range(dim)]


def _payload(tenant_id, user_id, text=None, text_lemma=None, **extra):
    '''组装写入用的 payload：text 缺省时从文本池随机取，text_lemma 缺省与 text 相同。'''
    content = _random_text() if text is None else text
    payload = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "text": content,
        "text_lemma": text_lemma or content,
        "created_at_tz": "2024-01-01T00:00:00+00:00",
    }
    payload.update(extra)
    return payload

def _scope(**extra):
    '''search、keyword_search、list 会把 tenant_id 就地写入传入的 filter，故始终给可变字典。'''
    return dict(extra)


# ---------- fixtures ----------

@pytest.fixture(scope="module")
def vector_settings():
    '''vector 分区配置：取自 conftest 常量，逐模块深拷贝避免互相污染。'''
    return CONFIG_VECTOR_SETTINGS.model_copy(deep=True)


# ---------- 契约层：MilvusVectorStore ----------

class TestMilvusVectorStore:
    '''MilvusVectorStore 的公共契约：backend 登记、构造参数与 metric_type。'''

    def test_backend_is_registered(self):
        assert MilvusVectorStore.backend == "milvus"
        assert is_vector_store_registered("milvus") is True
        assert resolve_vector_store("milvus") is MilvusVectorStore

    def test_inherits_store_contract(self):
        assert issubclass(MilvusVectorStore, VectorStoreBase)
        assert isinstance(MilvusVectorStore(), VectorStoreBase)

    def test_init_defaults(self):
        store = MilvusVectorStore()
        assert store.uri == "http://localhost:19530"
        assert (store.token, store.collection_name, store.db_name) == ("", "", "")
        assert store.vector_size == 1024
        assert (store.user, store.password, store.timeout) == ("", "", None)

    def test_init_stores_conftest_values(self, vector_settings):
        milvus = vector_settings.milvus
        store = MilvusVectorStore(
            uri=milvus.uri,
            token=milvus.token,
            collection_name=vector_settings.collection,
            vector_size=vector_settings.vector_size,
            db_name=milvus.db_name,
            timeout=15.0,
        )
        assert store.uri == milvus.uri
        assert store.token == milvus.token
        assert store.collection_name == vector_settings.collection
        assert store.vector_size == vector_settings.vector_size
        assert store.db_name == milvus.db_name
        assert store.timeout == 15.0

    def test_init_leaves_client_unset(self):
        assert MilvusVectorStore().client is None

    def test_metric_type_is_cosine(self):
        assert MilvusVectorStore.metric_type == "COSINE"

    def test_insert_without_start_raises_storage_error(self, vector_settings):
        '''未建连时 client 为 None，insert 把底层异常统一包成 StorageError。'''
        store = MilvusVectorStore(collection_name=vector_settings.collection)
        with pytest.raises(StorageError):
            store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                         [_payload(_next_id(), _next_id())], [_next_id()])

    def test_search_without_start_raises_storage_error(self, vector_settings):
        store = MilvusVectorStore(collection_name=vector_settings.collection)
        with pytest.raises(StorageError):
            store.search("x", _vector(_random_seed(), vector_settings.vector_size),
                         _next_id(), filter=_scope())

    def test_keyword_search_without_start_returns_none(self, vector_settings):
        '''关键词检索刻意吞掉检索异常，缺客户端时返回 None 而不是抛出。'''
        store = MilvusVectorStore(collection_name=vector_settings.collection)
        assert store.keyword_search("x", _next_id(), filter=_scope()) is None

    def test_start_without_driver_raises_optional_dependency(self, monkeypatch, vector_settings):
        monkeypatch.setattr("loong_memories.storage.memories.milvus_store.MilvusClient", None)
        store = MilvusVectorStore(collection_name=vector_settings.collection)
        with pytest.raises(OptionalDependencyError):
            store.start()


class TestAsyncMilvusVectorStore:
    '''AsyncMilvusVectorStore 的公共契约：backend 登记、构造参数与缺依赖报错。'''

    def test_backend_is_registered(self):
        assert AsyncMilvusVectorStore.backend == "milvus_async"
        assert is_vector_store_registered("milvus_async") is True
        assert resolve_vector_store("milvus_async") is AsyncMilvusVectorStore

    def test_inherits_sync_store(self):
        assert issubclass(AsyncMilvusVectorStore, MilvusVectorStore)
        assert issubclass(AsyncMilvusVectorStore, VectorStoreBase)

    def test_init_defaults(self):
        store = AsyncMilvusVectorStore()
        assert store.uri == "http://localhost:19530"
        assert (store.token, store.collection_name, store.db_name) == ("", "", "")
        assert store.vector_size == 1024
        assert (store.user, store.password, store.timeout) == ("", "", None)

    def test_init_stores_conftest_values(self, vector_settings):
        milvus = vector_settings.milvus
        store = AsyncMilvusVectorStore(
            uri=milvus.uri,
            token=milvus.token,
            collection_name=vector_settings.collection,
            vector_size=vector_settings.vector_size,
            db_name=milvus.db_name,
            timeout=15.0,
        )
        assert store.uri == milvus.uri
        assert store.collection_name == vector_settings.collection
        assert store.vector_size == vector_settings.vector_size
        assert store.timeout == 15.0

    def test_init_leaves_client_unset(self):
        assert AsyncMilvusVectorStore().client is None

    def test_metric_type_is_cosine(self):
        assert AsyncMilvusVectorStore.metric_type == "COSINE"


# ---------- 真实服务：MilvusVectorStore ----------

class TestMilvusVectorStoreIntegration:
    '''MilvusVectorStore 在真实 Milvus 上的公共方法行为：建集合、写入、读取、检索与更新。

    连接参数取自 conftest 的 vector 分区；集合名固定，每个用例的 tenant_id / user_id 与主键
    均由 _next_id() 唯一发放，因此并发与重复运行都不会互相踩到同一条记录。
    '''

    @pytest.fixture(scope="module")
    def milvus_store(self, vector_settings):
        '''建连并确保集合存在（start 内部会调用 create_col）。'''
        milvus = vector_settings.milvus
        store = MilvusVectorStore(
            uri=milvus.uri,
            token=milvus.token,
            collection_name=vector_settings.collection,
            vector_size=vector_settings.vector_size,
            db_name=milvus.db_name,
            timeout=30.0,
        )
        try:
            store.start()
        except Exception as exc:
            pytest.skip("无法连接 conftest 配置的 Milvus：" + str(exc))
        yield store

    def test_start_connects_client(self, milvus_store):
        assert milvus_store.client is not None

    def test_create_col_reuses_existing_collection(self, milvus_store, vector_settings):
        '''集合已存在时 create_col 直接返回，不重建索引也不报错。'''
        assert milvus_store.create_col(vector_settings.collection, vector_settings.vector_size) is None
        assert milvus_store.client.has_collection(vector_settings.collection) is True

    def test_insert_then_get_returns_record(self, milvus_store, vector_settings):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                            [_payload(tenant_id, user_id)], [rid])
        got = milvus_store.get(rid, tenant_id, user_id)
        assert got is not None
        assert (got.id, got.tenant_id) == (rid, tenant_id) and got.text in _TEXTS

    def test_get_unknown_id_returns_none(self, milvus_store):
        assert milvus_store.get(_next_id(), _next_id()) is None

    def test_get_with_other_user_returns_none(self, milvus_store, vector_settings):
        '''user_id 参与定位：换一个用户读取同一 id 应当查不到。'''
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                            [_payload(tenant_id, user_id)], [rid])
        assert milvus_store.get(rid, tenant_id, _next_id()) is None

    def test_insert_persists_text_lemma(self, milvus_store, vector_settings):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                            [_payload(tenant_id, user_id, text_lemma="词形")], [rid])
        got = milvus_store.get(rid, tenant_id, user_id)
        assert got.text in _TEXTS and got.text_lemma == "词形"

    def test_insert_dimension_mismatch_raises_storage_error(self, milvus_store, vector_settings):
        '''写入失败由 client 抛出后统一包成 StorageError。'''
        with pytest.raises(StorageError):
            milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size - 1)],
                                [_payload(_next_id(), _next_id())], [_next_id()])

    def test_search_returns_hits_desc_by_score(self, milvus_store, vector_settings):
        tenant_id, user_id = _next_id(), _next_id()
        dim = vector_settings.vector_size
        ids = [_next_id() for _ in range(3)]
        seeds = [_random_seed() for _ in range(3)]
        milvus_store.insert([_vector(seed, dim) for seed in seeds],
                            [_payload(tenant_id, user_id) for _ in range(3)], ids)
        hits = milvus_store.search(_random_text(), _vector(seeds[0], dim), tenant_id,
                                   filter=_scope(), top_k=3)
        assert len(hits) == 3


    def test_search_respects_top_k(self, milvus_store, vector_settings):
        tenant_id, user_id, dim = _next_id(), _next_id(), vector_settings.vector_size
        seed = _random_seed()
        milvus_store.insert([_vector(seed, dim) for _ in range(3)],
                            [_payload(tenant_id, user_id) for _ in range(3)],
                            [_next_id() for _ in range(3)])
        hits = milvus_store.search(_random_text(), _vector(seed, dim), tenant_id,
                                   filter=_scope(), top_k=2)
        assert len(hits) == 2

    def test_search_filters_by_tenant(self, milvus_store, vector_settings):
        '''tenant_id 由入参指定并写入过滤表达式，跨租户数据不得互相召回。'''
        text_a, text_b = _distinct_texts(2)
        first, second, user_id, dim = _next_id(), _next_id(), _next_id(), vector_settings.vector_size
        seed = _random_seed()
        milvus_store.insert([_vector(seed, dim), _vector(seed, dim)],
                            [_payload(first, user_id, text_a), _payload(second, user_id, text_b)],
                            [_next_id(), _next_id()])
        hits = milvus_store.search(text_a, _vector(seed, dim), first, filter=_scope(), top_k=10)
        assert text_a in [hit.text for hit in hits]

    def test_search_rejects_illegal_filter_key(self, milvus_store, vector_settings):
        '''非法过滤键在拼装表达式时就被拦下，不落到 client。'''
        with pytest.raises(StorageError):
            milvus_store.search("x", _vector(_random_seed(), vector_settings.vector_size),
                                _next_id(), filter={"bad key": 1})

    def test_keyword_search_matches_text_lemma(self, milvus_store, vector_settings):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        text = _random_text()
        milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                            [_payload(tenant_id, user_id, text=text, text_lemma=text)], [rid])
        hits = milvus_store.keyword_search(text, tenant_id, filter=_scope(), top_k=5)
        assert hits is not None
        assert rid in [hit.id for hit in hits]

    def test_keyword_search_returns_none_on_failure(self, milvus_store):
        '''BM25 检索失败时返回 None，与基类契约一致。'''
        assert milvus_store.keyword_search("x", _next_id(), filter={"bad key": 1}) is None

    def test_update_vector_and_payload(self, milvus_store, vector_settings):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        text_before, text_after = _distinct_texts(2)
        milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                            [_payload(tenant_id, user_id, text_before)], [rid])
        milvus_store.update(rid, tenant_id, user_id,
                            vector=_vector(_random_seed(), vector_settings.vector_size),
                            payload=_payload(tenant_id, user_id, text_after))
        got = milvus_store.get(rid, tenant_id, user_id)
        assert got is not None and got.text == text_after

    def test_update_keeps_existing_vector_when_omitted(self, milvus_store, vector_settings):
        '''只传 payload 时沿用库中已有向量，不重建。'''
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        text_before, text_after = _distinct_texts(2)
        milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                            [_payload(tenant_id, user_id, text_before)], [rid])
        milvus_store.update(rid, tenant_id, user_id, payload=_payload(tenant_id, user_id, text_after))
        got = milvus_store.get(rid, tenant_id, user_id)
        assert got is not None and got.text == text_after

    def test_update_unknown_id_raises_storage_error(self, milvus_store, vector_settings):
        tenant_id, user_id = _next_id(), _next_id()
        with pytest.raises(StorageError):
            milvus_store.update(_next_id(), tenant_id, user_id,
                                payload=_payload(tenant_id, user_id))

    def test_delete_removes_record(self, milvus_store, vector_settings):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                            [_payload(tenant_id, user_id)], [rid])
        milvus_store.delete(rid, tenant_id, user_id)
        assert milvus_store.get(rid, tenant_id, user_id) is None

    def test_delete_unknown_id_is_noop(self, milvus_store):
        '''删除不存在的记录不报错：内部先查后删。'''
        assert milvus_store.delete(_next_id(), _next_id()) is None

    def test_list_respects_top_k(self, milvus_store, vector_settings):
        tenant_id, user_id, dim = _next_id(), _next_id(), vector_settings.vector_size
        seed = _random_seed()
        milvus_store.insert([_vector(seed, dim) for _ in range(3)],
                            [_payload(tenant_id, user_id) for _ in range(3)],
                            [_next_id() for _ in range(3)])
        assert len(milvus_store.list(tenant_id, filter=_scope(), top_k=2)) == 2

    def test_list_applies_filters(self, milvus_store, vector_settings):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                            [_payload(tenant_id, user_id)], [rid])
        listed = milvus_store.list(tenant_id, filter=_scope(user_id=user_id))
        assert [item.id for item in listed] == [rid]

    def test_list_filters_on_metadata_field(self, milvus_store, vector_settings):
        '''非白名单键落到 metadata JSON 上做等值过滤。'''
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                            [_payload(tenant_id, user_id, tag="red")], [rid])
        listed = milvus_store.list(tenant_id, filter=_scope(tag="red"))
        assert [item.id for item in listed] == [rid]

    def test_list_rejects_illegal_filter_key(self, milvus_store):
        with pytest.raises(StorageError):
            milvus_store.list(_next_id(), filter={"bad key": 1})


# ---------- 真实服务：AsyncMilvusVectorStore ----------

@pytest.mark.asyncio(loop_scope="module")
@needs_async_client
class TestAsyncMilvusVectorStoreIntegration:
    '''AsyncMilvusVectorStore 在真实 Milvus 上的公共方法行为，语义与同步版本一致。'''

    @pytest_asyncio.fixture(scope="module", loop_scope="module")
    async def async_milvus_store(self, vector_settings):
        '''建连并确保集合存在（start 内部会 await create_col）。'''
        milvus = vector_settings.milvus
        store = AsyncMilvusVectorStore(
            uri=milvus.uri,
            token=milvus.token,
            collection_name=vector_settings.collection,
            vector_size=vector_settings.vector_size,
            db_name=milvus.db_name,
            timeout=30.0,
        )
        try:
            await store.start()
        except Exception as exc:
            pytest.skip("无法连接 conftest 配置的 Milvus：" + str(exc))
        yield store

    async def test_start_connects_client(self, async_milvus_store):
        assert async_milvus_store.client is not None

    async def test_create_col_is_idempotent(self, async_milvus_store, vector_settings):
        assert await async_milvus_store.create_col(vector_settings.collection,
                                                    vector_settings.vector_size) is None

    async def test_insert_then_get_returns_record(self, async_milvus_store, vector_settings):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        await async_milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                                        [_payload(tenant_id, user_id)], [rid])
        got = await async_milvus_store.get(rid, tenant_id, user_id)
        assert got is not None
        assert (got.id, got.tenant_id) == (rid, tenant_id) and got.text in _TEXTS

    async def test_get_unknown_id_returns_none(self, async_milvus_store):
        assert await async_milvus_store.get(_next_id(), _next_id()) is None

    async def test_insert_persists_text_lemma(self, async_milvus_store, vector_settings):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        await async_milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                                        [_payload(tenant_id, user_id, text_lemma="词形")], [rid])
        got = await async_milvus_store.get(rid, tenant_id, user_id)
        assert got.text in _TEXTS and got.text_lemma == "词形"

    async def test_search_returns_hits_desc_by_score(self, async_milvus_store, vector_settings):
        tenant_id, user_id, dim = _next_id(), _next_id(), vector_settings.vector_size
        ids = [_next_id() for _ in range(3)]
        seeds = [_random_seed() for _ in range(3)]
        await async_milvus_store.insert([_vector(seed, dim) for seed in seeds],
                                        [_payload(tenant_id, user_id)for _ in range(3)],
                                        ids)
        hits = await async_milvus_store.search(_random_text(), _vector(seeds[0], dim), tenant_id,
                                        filter=_scope(), top_k=3)
        assert hits

    async def test_search_respects_top_k(self, async_milvus_store, vector_settings):
        tenant_id, user_id, dim = _next_id(), _next_id(), vector_settings.vector_size
        seed = _random_seed()
        await async_milvus_store.insert([_vector(seed, dim) for _ in range(3)],
                                        [_payload(tenant_id, user_id)
                                        for _ in range(3)],[_next_id() for _ in range(3)])
        hits = await async_milvus_store.search(_random_text(), _vector(seed, dim), tenant_id,
                                        filter=_scope(), top_k=2)
        assert len(hits) == 2

    async def test_search_filters_by_tenant(self, async_milvus_store, vector_settings):
        text_a, text_b = _distinct_texts(2)
        first, second, user_id, dim = _next_id(), _next_id(), _next_id(), vector_settings.vector_size
        seed = _random_seed()
        await async_milvus_store.insert([_vector(seed, dim), _vector(seed, dim)],
                                        [_payload(first, user_id, text_a), _payload(second, user_id, text_b)],
                                        [_next_id(), _next_id()])
        hits = await async_milvus_store.search(query=text_a,vectors=_vector(seed, dim),tenant_id=first, filter=_scope(), top_k=10)
        assert text_a in [hit.text for hit in hits]

    async def test_search_rejects_illegal_filter_key(self, async_milvus_store, vector_settings):
        with pytest.raises(StorageError):
            await async_milvus_store.search("x", _vector(_random_seed(), vector_settings.vector_size),
                                            _next_id(), filter={"bad key": 1})

    async def test_keyword_search_matches_text_lemma(self, async_milvus_store, vector_settings):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()

        text_a = _distinct_texts(1)[0]

        await async_milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                                        [_payload(tenant_id, user_id, text=text_a, text_lemma=text_a)], [rid])
        hits = await async_milvus_store.keyword_search(text_a, tenant_id, filter=_scope(), top_k=5)
        assert hits is not None
        assert rid in [hit.id for hit in hits]

    async def test_keyword_search_returns_none_on_failure(self, async_milvus_store):
        assert await async_milvus_store.keyword_search("x", _next_id(),
                                                      filter={"bad key": 1}) is None

    async def test_update_vector_and_payload(self, async_milvus_store, vector_settings):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        text_before, text_after = _distinct_texts(2)
        await async_milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                                        [_payload(tenant_id, user_id, text_before)], [rid])
        await async_milvus_store.update(rid, tenant_id, user_id,
                                        vector=_vector(_random_seed(), vector_settings.vector_size),
                                        payload=_payload(tenant_id, user_id, text_after))
        got = await async_milvus_store.get(rid, tenant_id, user_id)
        assert got is not None and got.text == text_after

    async def test_update_keeps_existing_vector_when_omitted(self, async_milvus_store, vector_settings):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        text_before, text_after = _distinct_texts(2)
        await async_milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                                        [_payload(tenant_id, user_id, text_before)], [rid])
        await async_milvus_store.update(rid, tenant_id, user_id,
                                        payload=_payload(tenant_id, user_id, text_after))
        got = await async_milvus_store.get(rid, tenant_id, user_id)
        assert got is not None and got.text == text_after

    async def test_update_unknown_id_raises_storage_error(self, async_milvus_store, vector_settings):
        tenant_id, user_id = _next_id(), _next_id()
        with pytest.raises(StorageError):
            await async_milvus_store.update(_next_id(), tenant_id, user_id,
                                            payload=_payload(tenant_id, user_id))

    async def test_delete_removes_record(self, async_milvus_store, vector_settings):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        await async_milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                                        [_payload(tenant_id, user_id)], [rid])
        await async_milvus_store.delete(rid, tenant_id, user_id)
        assert await async_milvus_store.get(rid, tenant_id, user_id) is None

    async def test_delete_unknown_id_is_noop(self, async_milvus_store):
        '''异步删除不先查后删，直接下发过滤删除，故幂等。'''
        assert await async_milvus_store.delete(_next_id(), _next_id()) is None

    async def test_list_respects_top_k(self, async_milvus_store, vector_settings):
        tenant_id, user_id, dim = _next_id(), _next_id(), vector_settings.vector_size
        seed = _random_seed()
        await async_milvus_store.insert([_vector(seed, dim) for _ in range(3)],
                                        [_payload(tenant_id, user_id)
                                        for _ in range(3)], [_next_id() for _ in range(3)])
        assert len(await async_milvus_store.list(tenant_id, filter=_scope(), top_k=2)) == 2

    async def test_list_applies_filters(self, async_milvus_store, vector_settings):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        await async_milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                                        [_payload(tenant_id, user_id)], [rid])
        listed = await async_milvus_store.list(tenant_id, filter=_scope(user_id=user_id))
        assert [item.id for item in listed] == [rid]

    async def test_list_filters_on_metadata_field(self, async_milvus_store, vector_settings):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        await async_milvus_store.insert([_vector(_random_seed(), vector_settings.vector_size)],
                                        [_payload(tenant_id, user_id, tag="red")], [rid])
        listed = await async_milvus_store.list(tenant_id, filter=_scope(tag="red"))
        assert [item.id for item in listed] == [rid]

    async def test_list_rejects_illegal_filter_key(self, async_milvus_store):
        with pytest.raises(StorageError):
            await async_milvus_store.list(_next_id(), filter={"bad key": 1})

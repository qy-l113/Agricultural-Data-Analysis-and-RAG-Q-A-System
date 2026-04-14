# app.py - 支持玉米和小麦双作物预测

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
import dotenv
from openai import OpenAI

# 设置离线模式（禁止联网请求）
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.RandomForestRegressor import YieldPredictorRF as YieldPredictor

# 加载环境变量
dotenv.load_dotenv()

# 页面配置
st.set_page_config(
    page_title="农业智能问答系统",
    page_icon="🌾",
    layout="wide"
)


# ========== 初始化函数 ==========

@st.cache_resource
def init_maize_predictor():
    """初始化玉米产量预测模型"""
    predictor = YieldPredictor(data_root='./data', crop='maize')
    predictor.load_all_data()
    predictor.preprocess_data()
    predictor.load_model('./models/saved_models/maize_randomforest.pkl')
    return predictor


@st.cache_resource
def init_wheat_predictor():
    """初始化小麦产量预测模型"""
    predictor = YieldPredictor(data_root='./data', crop='wheat')
    predictor.load_all_data()
    predictor.preprocess_data()
    predictor.load_model('./models/saved_models/wheat_randomforest.pkl')
    return predictor


@st.cache_resource
def get_client():
    """初始化 OpenAI 客户端"""
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        return OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    return None


@st.cache_resource
def get_retriever():
    """加载知识库检索器"""
    vectordb_path = './vectordb'
    if not os.path.exists(vectordb_path):
        st.info("📚 知识库未构建，将使用纯AI问答模式")
        return None

    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="./models/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        vectordb = Chroma(
            persist_directory=vectordb_path,
            embedding_function=embeddings
        )

        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        return retriever
    except Exception as e:
        st.warning(f"知识库加载失败: {e}")
        return None


# ========== 辅助函数 ==========

def get_region_list(predictor):
    """获取可预测的地区列表"""
    location_map = {
        'CN011': '北京', 'CN012': '天津', 'CN013': '河北', 'CN014': '山西',
        'CN015': '内蒙古', 'CN021': '辽宁', 'CN022': '吉林', 'CN023': '黑龙江',
        'CN031': '上海', 'CN032': '江苏', 'CN033': '浙江', 'CN034': '安徽',
        'CN035': '福建', 'CN036': '江西', 'CN037': '山东', 'CN041': '河南',
        'CN042': '湖北', 'CN043': '湖南', 'CN044': '广东', 'CN045': '广西',
        'CN046': '海南', 'CN050': '重庆', 'CN051': '四川', 'CN052': '贵州',
        'CN053': '云南', 'CN054': '西藏', 'CN061': '陕西', 'CN062': '甘肃',
        'CN063': '青海', 'CN064': '宁夏', 'CN065': '新疆', 'CN071': '台湾'
    }

    regions = []
    for adm_id, name in location_map.items():
        if adm_id in predictor.location_df['adm_id'].values:
            regions.append(name)
    return regions


def get_yield_prediction(region, year, crop, maize_predictor, wheat_predictor):
    """根据作物类型进行产量预测"""
    code_map = {v: k for k, v in {
        '北京': 'CN011', '天津': 'CN012', '河北': 'CN013', '山西': 'CN014',
        '内蒙古': 'CN015', '辽宁': 'CN021', '吉林': 'CN022', '黑龙江': 'CN023',
        '上海': 'CN031', '江苏': 'CN032', '浙江': 'CN033', '安徽': 'CN034',
        '福建': 'CN035', '江西': 'CN036', '山东': 'CN037', '河南': 'CN041',
        '湖北': 'CN042', '湖南': 'CN043', '广东': 'CN044', '广西': 'CN045',
        '海南': 'CN046', '重庆': 'CN050', '四川': 'CN051', '贵州': 'CN052',
        '云南': 'CN053', '西藏': 'CN054', '陕西': 'CN061', '甘肃': 'CN062',
        '青海': 'CN063', '宁夏': 'CN064', '新疆': 'CN065', '台湾': 'CN071'
    }.items()}

    adm_id = code_map.get(region, 'CN032')

    # 选择对应的预测器
    if crop == "玉米":
        predictor = maize_predictor
        crop_name = "玉米"
    else:
        predictor = wheat_predictor
        crop_name = "小麦"

    result = predictor.predict(adm_id, year)

    if result['success']:
        return {
            'success': True,
            'crop': crop_name,
            'predicted_yield': result['predicted_yield'],
            'base_year': result['base_year'],
            'base_yield': result['base_yield']
        }
    return {'success': False}


def ask_llm(question, client, retriever=None):
    """调用 LLM 回答问题"""
    if client is None:
        return "抱歉，AI 服务未配置。请检查 API 密钥设置。"

    context = ""
    if retriever is not None:
        try:
            docs = retriever.invoke(question)
            if docs:
                context = "\n\n【📚 小农的知识库参考】\n"
                for i, doc in enumerate(docs):
                    content = doc.page_content[:500]
                    context += f"{i + 1}. {content}\n"
                context += "\n请基于以上知识库内容回答用户问题，并在回答中体现你参考了知识库。\n"
        except Exception as e:
            print(f"知识库检索失败: {e}")

    try:
        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": f"""你是专业的农业助手，名字叫"小农"。

【你的身份】
- 你是一个亲切、专业的农业助手
- 名字叫"小农"，喜欢用🌾表情
- 回答要像朋友聊天一样自然

【回答规则】
1. **必须以"🌾 小农："开头**
2. 如果参考了知识库，请在回答开头加上"📚 根据小农的知识库，"
3. 如果回答涉及预测数据，加上"📊 根据预测模型，"
4. 用通俗易懂的话解释专业概念
5. 回答简洁但信息完整

{context}

记住：你是小农，不是普通的AI助手！每次回答都要以"🌾 小农："开头。"""},
                {"role": "user", "content": question}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"抱歉，AI服务调用失败: {str(e)}"


def process_question(question, maize_predictor, wheat_predictor, client, retriever=None):
    """处理用户问题，自动识别作物类型"""

    # 判断是什么作物
    crop_type = None
    if any(kw in question for kw in ['玉米', 'maize', 'corn']):
        crop_type = "玉米"
        predictor = maize_predictor
    elif any(kw in question for kw in ['小麦', 'wheat']):
        crop_type = "小麦"
        predictor = wheat_predictor

    # 判断是否是产量预测问题
    is_prediction = any(kw in question for kw in ['预测', '产量', '预计', '收成', '亩产'])
    is_location = any(p in question for p in
                      ['江苏', '山东', '河南', '河北', '安徽', '四川', '北京', '天津', '吉林', '辽宁', '黑龙江'])

    # 如果是预测问题且有地区信息和作物信息
    if is_prediction and is_location and crop_type:
        province = None
        for p in ['江苏', '山东', '河南', '河北', '安徽', '四川', '北京', '天津', '吉林', '辽宁', '黑龙江']:
            if p in question:
                province = p
                break
        if province:
            result = get_yield_prediction(province, 2024, crop_type, maize_predictor, wheat_predictor)
            if result['success']:
                return f"""🌾 **小农预测**：{province}地区2024年{result['crop']}产量预计为 **{result['predicted_yield']} 吨/公顷**（约 {result['predicted_yield'] * 66.7:.1f} 公斤/亩）。

📊 基于 {result['base_year']} 年产量 {result['base_yield']} 吨/公顷的历史数据。

💡 实际产量可能受天气影响，建议关注气象预报。"""
            else:
                return f"抱歉，暂时无法获取{province}的{crop_type}产量预测数据。"

    # 使用带知识库的 LLM 回答
    return ask_llm(question, client, retriever)


# ========== 主程序 ==========

# 初始化
with st.spinner("正在加载系统..."):
    maize_predictor = init_maize_predictor()
    wheat_predictor = init_wheat_predictor()
    client = get_client()
    retriever = get_retriever()

    # 获取地区列表（用玉米的）
    regions = get_region_list(maize_predictor)

# 侧边栏
with st.sidebar:
    st.title("🌾 农业数据看板")
    st.markdown("---")

    st.markdown("### 📊 快速预测")

    # 作物选择
    crop_option = st.selectbox("作物类型", ["玉米", "小麦"], key="crop_select")

    # 地区选择
    region = st.selectbox("省份", regions, key="region_select")

    # 年份选择
    year = st.number_input("年份", min_value=2024, max_value=2030, value=2024)

    if st.button("🔮 预测产量", type="primary"):
        result = get_yield_prediction(region, year, crop_option, maize_predictor, wheat_predictor)
        if result['success']:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%); 
                        color: white; padding: 20px; border-radius: 10px; text-align: center;'>
                <h3>{region} {year}年{result['crop']}产量</h3>
                <p style='font-size: 32px; font-weight: bold; margin: 10px 0;'>
                    {result['predicted_yield']} 吨/公顷
                </p>
                <p>≈ {result['predicted_yield'] * 66.7:.1f} 公斤/亩</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # 显示系统状态
    status_msg = f"""
    **系统信息**
    - 玉米模型: {'✅ 已加载' if maize_predictor.model else '❌ 未加载'}
    - 小麦模型: {'✅ 已加载' if wheat_predictor.model else '❌ 未加载'}
    - AI 问答: {'✅ 已启用' if client else '❌ 未启用'}
    - 知识库: {'✅ 已加载' if retriever else '❌ 未加载'}
    """
    st.info(status_msg)

    st.markdown("---")
    st.caption("📚 数据来源：国家统计局 | 中国气象数据网 | 遥感监测")

# 主界面
st.title("🌾 农业智能问答系统")
st.caption("基于机器学习产量预测 + RAG知识库检索 + AI智能问答")

# 创建两列布局
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 智能问答")

    # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "🌾 小农：您好！我是农业智能助手\"小农\"，可以帮您：\n\n• 📊 预测玉米/小麦产量\n• 💡 解答农业技术问题\n• 🌡️ 分析气象对产量的影响\n• 📚 从知识库检索专业知识\n\n请问有什么可以帮您？"}
        ]

    # 显示聊天记录
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 用户输入
    if prompt := st.chat_input("输入您的农业问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🌾 小农正在思考..."):
                response = process_question(prompt, maize_predictor, wheat_predictor, client, retriever)
                st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

with col2:
    st.subheader("💡 推荐问题")

    # 根据侧边栏选择的作物显示不同的推荐问题
    if crop_option == "玉米":
        recommended = [
            "🌽 玉米什么时候播种？",
            "🐛 玉米螟怎么防治？",
            "📊 山东玉米今年产量预测多少？",
            "🛰️ 什么是NDVI？",
            "💧 玉米需要多少水？",
            "🌱 玉米施肥要注意什么？"
        ]
    else:
        recommended = [
            "🌾 小麦什么时候播种？",
            "🐛 小麦锈病怎么防治？",
            "📊 河南小麦今年产量预测多少？",
            "🌱 小麦施肥技术要点？",
            "💧 小麦浇冻水什么时候？",
            "🌾 强筋小麦和弱筋小麦有什么区别？"
        ]

    # 使用会话状态记录选中的问题
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = None

    for q in recommended:
        if st.button(q, use_container_width=True, key=f"btn_{q[:15]}"):
            st.session_state.selected_question = q

    # 处理选中的问题
    if st.session_state.selected_question:
        question = st.session_state.selected_question
        st.session_state.selected_question = None

        st.session_state.messages.append({"role": "user", "content": question})
        response = process_question(question, maize_predictor, wheat_predictor, client, retriever)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    st.markdown("---")
    st.subheader("📈 产量趋势")

    # 作物选择
    trend_crop = st.radio("选择作物", ["玉米", "小麦"], horizontal=True, key="trend_crop")

    if trend_crop == "玉米":
        trend_predictor = maize_predictor
    else:
        trend_predictor = wheat_predictor

    selected = st.selectbox("选择地区", regions[:8], key="trend_select")
    code_map_rev = {v: k for k, v in {
        '北京': 'CN011', '天津': 'CN012', '河北': 'CN013', '山西': 'CN014',
        '内蒙古': 'CN015', '辽宁': 'CN021', '吉林': 'CN022', '黑龙江': 'CN023',
        '上海': 'CN031', '江苏': 'CN032', '浙江': 'CN033', '安徽': 'CN034',
        '福建': 'CN035', '江西': 'CN036', '山东': 'CN037', '河南': 'CN041',
    }.items()}

    adm_id = code_map_rev.get(selected, 'CN032')

    # 获取历史数据
    if hasattr(trend_predictor, 'df') and trend_predictor.df is not None:
        history = trend_predictor.df[trend_predictor.df['adm_id'] == adm_id].sort_values('year')
        if len(history) > 0:
            fig = px.line(history, x='year', y='yield',
                          title=f'{selected}{trend_crop}产量趋势',
                          labels={'year': '年份', 'yield': '产量(吨/公顷)'})
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"暂无{selected}{trend_crop}历史数据")
    else:
        st.info("请先训练小麦模型")

st.markdown("---")
st.caption("📚 数据来源：国家统计局 | 中国气象数据网 | 遥感监测 | 小农知识库")
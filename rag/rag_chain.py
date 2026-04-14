# rag/rag_chain.py

import os
import sys
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.RandomForestRegressor import YieldPredictorRF as YieldPredictor


class AgricultureRAGSystem:
    """农业智能问答系统 - 整合RAG和产量预测（支持玉米和小麦）"""

    def __init__(self, data_root='./data'):
        """
        初始化RAG系统
        """
        # 加载环境变量
        dotenv.load_dotenv()

        # ========== 初始化玉米产量预测模型 ==========
        print("正在加载玉米产量预测模型...")
        self.maize_predictor = YieldPredictor(data_root=data_root, crop='maize')
        print("加载玉米数据...")
        self.maize_predictor.load_all_data()
        print("预处理玉米数据...")
        self.maize_predictor.preprocess_data(use_lag_features=False)

        maize_model_path = './models/saved_models/maize_randomforest.pkl'
        if os.path.exists(maize_model_path):
            self.maize_predictor.load_model(maize_model_path)
            print("✓ 玉米产量预测模型加载成功")
        else:
            print("⚠️ 玉米模型文件不存在，需要先训练模型")

        # ========== 初始化小麦产量预测模型 ==========
        print("\n正在加载小麦产量预测模型...")
        self.wheat_predictor = YieldPredictor(data_root=data_root, crop='wheat')
        print("加载小麦数据...")
        self.wheat_predictor.load_all_data()
        print("预处理小麦数据...")
        self.wheat_predictor.preprocess_data(use_lag_features=False)

        wheat_model_path = './models/saved_models/wheat_randomforest.pkl'
        if os.path.exists(wheat_model_path):
            self.wheat_predictor.load_model(wheat_model_path)
            print("✓ 小麦产量预测模型加载成功")
        else:
            print("⚠️ 小麦模型文件不存在，需要先训练模型")

        # 确保 location_df 存在（使用玉米的）
        if not hasattr(self.maize_predictor, 'location_df') or self.maize_predictor.location_df is None:
            print("⚠️ location_df 不存在，尝试重新加载...")
            import pandas as pd
            location_path = os.path.join(data_root, 'CN', 'maize', 'location_maize_CN.csv')
            if os.path.exists(location_path):
                self.maize_predictor.location_df = pd.read_csv(location_path)
                print("✓ location_df 加载成功")

        # ========== 初始化大语言模型 ==========
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DEEPSEEK_API_KEY")

        if not api_key:
            print("⚠️ 未找到API密钥，将使用模拟模式")
            print("请在 .env 文件中设置 DASHSCOPE_API_KEY 或 DEEPSEEK_API_KEY")
            self.llm = None
        else:
            try:
                self.llm = ChatOpenAI(
                    temperature=0.7,
                    model="qwen-turbo",
                    openai_api_key=api_key,
                    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    streaming=True
                )
                print("✓ 大语言模型初始化成功")
            except Exception as e:
                print(f"⚠️ 大语言模型初始化失败: {e}")
                self.llm = None

        # 初始化对话历史
        self.chat_history = []

        # 构建RAG链
        self._build_chain()

        print("✓ RAG系统初始化完成")

    def _build_chain(self):
        """构建RAG链"""
        system_prompt = """你是一个专业的农业智能助手，名叫"小农"。你的职责是：

1. **产量预测**：当用户询问产量预测时，调用预测模型给出具体数值
2. **农业知识问答**：回答作物种植、病虫害防治、施肥灌溉等农业技术问题
3. **数据分析**：解释产量变化趋势，分析影响因素

回答要求：
- 专业、准确、通俗易懂
- 涉及预测时给出具体数据
- 不确定的信息要明确说明
- 回答简洁，但信息完整

{context}"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        if self.llm is None:
            self.rag_chain = None
        else:
            self.rag_chain = self.prompt | self.llm | StrOutputParser()

    def _get_predictor_by_crop(self, crop):
        """根据作物类型获取对应的预测器"""
        if crop == "玉米":
            return self.maize_predictor
        elif crop == "小麦":
            return self.wheat_predictor
        else:
            return self.maize_predictor  # 默认返回玉米

    def _is_prediction_question(self, question):
        """判断是否是产量预测相关的问题"""
        prediction_keywords = [
            '预测', '产量', '预计', '估计', '收成', '亩产',
            '今年产量', '明年产量', '产量多少', '能产多少',
            'forecast', 'predict', 'yield'
        ]
        return any(kw in question for kw in prediction_keywords)

    def _extract_crop_from_question(self, question):
        """从问题中提取作物类型"""
        if '玉米' in question or 'maize' in question.lower() or 'corn' in question.lower():
            return '玉米'
        elif '小麦' in question or 'wheat' in question.lower():
            return '小麦'
        return None

    def _extract_location_from_question(self, question):
        """从问题中提取地区信息"""
        location_map = {
            '北京': 'CN011', '天津': 'CN012', '河北': 'CN013', '山西': 'CN014',
            '内蒙古': 'CN015', '辽宁': 'CN021', '吉林': 'CN022', '黑龙江': 'CN023',
            '上海': 'CN031', '江苏': 'CN032', '浙江': 'CN033', '安徽': 'CN034',
            '福建': 'CN035', '江西': 'CN036', '山东': 'CN037', '河南': 'CN041',
            '湖北': 'CN042', '湖南': 'CN043', '广东': 'CN044', '广西': 'CN045',
            '海南': 'CN046', '重庆': 'CN050', '四川': 'CN051', '贵州': 'CN052',
            '云南': 'CN053', '西藏': 'CN054', '陕西': 'CN061', '甘肃': 'CN062',
            '青海': 'CN063', '宁夏': 'CN064', '新疆': 'CN065', '台湾': 'CN071'
        }
        for name, code in location_map.items():
            if name in question:
                return code, name
        return None, None

    def _get_prediction_answer(self, question):
        """处理产量预测问题"""
        # 提取作物类型
        crop = self._extract_crop_from_question(question)
        if crop is None:
            crop = "玉米"  # 默认玉米

        # 提取地区
        adm_id, province = self._extract_location_from_question(question)

        if adm_id is None:
            return f"请问您想查询哪个省份的{crop}产量预测？例如：江苏、山东、河南等。"

        # 获取对应的预测器
        predictor = self._get_predictor_by_crop(crop)

        # 获取预测
        result = predictor.predict(adm_id, 2024)

        if not result['success']:
            return f"抱歉，暂时无法获取{province}的{crop}产量预测数据。"

        # 获取历史趋势
        if hasattr(predictor, 'df') and predictor.df is not None:
            history = predictor.df[predictor.df['adm_id'] == adm_id].sort_values('year')
            recent_years = history.tail(5)

            answer = f"""根据模型预测，**{province}地区2024年{crop}产量预计为 {result['predicted_yield']} 吨/公顷**（约 {result['predicted_yield'] * 66.7:.1f} 公斤/亩）。

📊 **历史参考**：
"""
            for _, row in recent_years.iterrows():
                answer += f"  • {int(row['year'])}年: {row['yield']:.2f} 吨/公顷\n"
        else:
            answer = f"""根据模型预测，**{province}地区2024年{crop}产量预计为 {result['predicted_yield']} 吨/公顷**（约 {result['predicted_yield'] * 66.7:.1f} 公斤/亩）。

💡 **分析说明**：
• 基于{result['base_year']}年产量 {result['base_yield']} 吨/公顷
"""

        answer += """
💡 **温馨提示**：实际产量可能受极端天气等因素影响。

如需更详细的分析，请提供具体的气象数据。"""
        return answer

    def query(self, question):
        """处理用户问题"""
        if self._is_prediction_question(question):
            return self._get_prediction_answer(question)

        if self.rag_chain is None:
            if self.llm is None:
                return "抱歉，LLM服务未配置。请检查API密钥设置。"
            else:
                return "抱歉，RAG链初始化失败。请检查系统配置。"

        try:
            response = self.rag_chain.invoke({
                "question": question,
                "chat_history": self.chat_history
            })
            self.chat_history.extend([
                HumanMessage(content=question),
                AIMessage(content=response)
            ])
            return response
        except Exception as e:
            return f"抱歉，处理您的问题时出现错误：{str(e)}"

    def get_available_regions(self):
        """获取可预测的地区列表"""
        regions = []
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

        if hasattr(self.maize_predictor, 'location_df') and self.maize_predictor.location_df is not None:
            available_ids = self.maize_predictor.location_df['adm_id'].values
            for adm_id, name in location_map.items():
                if adm_id in available_ids:
                    regions.append(name)
        else:
            regions = list(location_map.values())
        return regions


if __name__ == "__main__":
    print("正在初始化农业智能问答系统...")
    rag = AgricultureRAGSystem()
    print("\n" + "=" * 50)
    print("农业智能问答系统测试")
    print("=" * 50)
    test_questions = [
        "江苏玉米今年产量预测多少？",
        "河南小麦产量预计多少？"
    ]
    for q in test_questions:
        print(f"\n用户: {q}")
        print(f"助手: {rag.query(q)}")
        print("-" * 40)
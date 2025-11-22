# environmental_science_assistant.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from zhipuai import ZhipuAI
import gradio as gr
import re
from datetime import datetime
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

class EnvironmentalScienceAssistant:
    def __init__(self, api_key):
        """初始化环境科学AI助手"""
        self.client = ZhipuAI(api_key=api_key)
        self.df = None
        self.experiment_type = None
        self.analysis_history = []
        self.data_loaded = False  # 添加实例级别的加载状态
    def detect_experiment_type(self, df):
        """自动识别实验数据类型"""
        columns = [col.lower() for col in df.columns]
        
        # 水质检测相关关键词
        water_quality_keywords = ['cod', '六价铬', '铬', '水质', '水样', 'ph', '溶解氧', '浊度', '氨氮', '总磷']
        # 气候变化相关关键词  
        climate_keywords = ['温度', '气温', '降水', '湿度', 'co2', '碳', '温室气体', '水位', '海平面']
        # 生态监测相关关键词
        ecology_keywords = ['物种', '生物量', '多样性', '丰度', '基因组', 'dna', 'rna', '微生物']
        # 土壤分析相关关键词
        soil_keywords = ['土壤', '重金属', '养分', '氮', '磷', '钾', '有机质']
        
        detected_types = []
        
        if any(keyword in ' '.join(columns) for keyword in water_quality_keywords):
            detected_types.append(("水质检测", 0.8))
        if any(keyword in ' '.join(columns) for keyword in climate_keywords):
            detected_types.append(("气候变化", 0.7))
        if any(keyword in ' '.join(columns) for keyword in ecology_keywords):
            detected_types.append(("生态监测", 0.6))
        if any(keyword in ' '.join(columns) for keyword in soil_keywords):
            detected_types.append(("土壤分析", 0.6))
            
        if detected_types:
            # 返回置信度最高的类型
            detected_types.sort(key=lambda x: x[1], reverse=True)
            return detected_types[0][0]
        else:
            return "通用环境数据"
    
    def load_data(self, file_content, file_type="csv"):
        """加载环境科学实验数据"""
        try:
            if file_type == "csv":
                self.df = pd.read_csv(StringIO(file_content))
            else:
                self.df = pd.read_csv(StringIO(file_content))
            
            # 验证数据不为空
            if self.df.empty:
                return False, f"❌ 数据加载失败：文件为空或格式不正确"
            
            # 自动识别实验类型
            self.experiment_type = self.detect_experiment_type(self.df)
            
            # 数据预处理
            self._preprocess_data()
            
            return True, f"✅ 数据加载成功！识别为【{self.experiment_type}】实验\n📊 数据规模：{len(self.df)}行 × {len(self.df.columns)}列"
            
        except pd.errors.EmptyDataError:
            return False, f"❌ 数据加载失败：文件为空"
        except pd.errors.ParserError as e:
            return False, f"❌ CSV格式解析错误：{str(e)}"
        except Exception as e:
            return False, f"❌ 数据加载失败：{str(e)}"
    
    def _preprocess_data(self):
        """通用数据预处理"""
        # 处理时间列
        time_columns = [col for col in self.df.columns 
                       if any(word in col.lower() for word in ['时间', 'date', 'time', '采样时间'])]
        if time_columns:
            time_col = time_columns[0]
            try:
                self.df[time_col] = pd.to_datetime(self.df[time_col])
            except:
                pass
        
        # 自动识别数值列并进行基本清洗
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # 移除明显异常值（超过5倍标准差）
            mean_val = self.df[col].mean()
            std_val = self.df[col].std()
            if std_val > 0:
                self.df = self.df[np.abs(self.df[col] - mean_val) <= 5 * std_val]
    
    def analyze_experiment(self, user_query):
        """智能分析环境实验数据"""
        if self.df is None:
            return "请先加载实验数据", None
        
        try:
            # 根据实验类型选择不同的专业提示词
            prompt_templates = {
                "水质检测": self._get_water_quality_prompt(),
                "气候变化": self._get_climate_change_prompt(), 
                "生态监测": self._get_ecology_prompt(),
                "土壤分析": self._get_soil_analysis_prompt(),
                "通用环境数据": self._get_general_prompt()
            }
            
            system_prompt = prompt_templates.get(self.experiment_type, self._get_general_prompt())
            
            # 准备数据摘要
            data_summary = f"""
            【实验数据摘要】
            实验类型：{self.experiment_type}
            数据规模：{len(self.df)}行记录，{len(self.df.columns)}个参数
            时间范围：{self._get_time_range()}
            监测参数：{', '.join(self.df.columns)}
            
            【数据统计摘要】
            {self._get_detailed_statistics()}
            """
            
            full_prompt = f"{system_prompt}\n{data_summary}\n\n【用户分析需求】：{user_query}"
            
            # 调用智谱AI
            response = self.client.chat.completions.create(
                model="glm-4",
                messages=[{"role": "user", "content": full_prompt}],
            )
            
            ai_response = response.choices[0].message.content
            
            # 生成专业可视化
            plot_path = self._generate_professional_plot(user_query)
            
            # 记录分析历史
            self.analysis_history.append({
                'timestamp': datetime.now(),
                'experiment_type': self.experiment_type,
                'query': user_query,
                'response': ai_response[:200] + "..." if len(ai_response) > 200 else ai_response
            })
            
            return ai_response, plot_path
            
        except Exception as e:
            return f"分析过程中出现错误：{str(e)}", None
    
    def _get_water_quality_prompt(self):
        """水质检测专业提示词"""
        return """你是一名环境工程专家，专门分析水质检测数据。请根据水质监测数据提供专业分析。

专业知识要点：
- COD（化学需氧量）：衡量水中有机物污染程度，地表水Ⅰ类≤15mg/L，Ⅴ类≤40mg/L
- 六价铬：有毒重金属，饮用水标准≤0.05mg/L
- 水质综合评价：单因子评价法、内梅罗指数法等
- 污染源识别：工业废水、生活污水、农业面源等特征分析

请提供：
1. 水质参数达标情况评估
2. 污染程度分级评价  
3. 可能的污染源分析
4. 治理建议和监测方案"""
    
    def _get_climate_change_prompt(self):
        """气候变化分析提示词"""
        return """你是一名气候变化研究专家，擅长分析气候监测和全球变化实验数据。

专业知识要点：
- 温度变化趋势：线性回归分析显著性
- 极端气候事件：频率和强度变化
- 水位变化：与温度、降水的相关性
- 控制实验：增温、降水控制等实验设计原理

请提供：
1. 气候变化趋势分析
2. 环境因子相关性分析
3. 实验处理效应评估
4. 生态影响预测"""
    
    def _get_ecology_prompt(self):
        """生态监测分析提示词"""
        return """你是一名生态学专家，擅长生物多样性监测和宏基因组数据分析。

专业知识要点：
- α多样性：Shannon-Wiener指数、Simpson指数
- β多样性：群落相似性分析  
- 物种组成：优势种、关键种识别
- 宏基因组：功能基因注释、代谢通路分析

请提供：
1. 生物多样性评估
2. 群落结构分析
3. 环境驱动因子识别
4. 生态功能预测"""
    
    def _get_soil_analysis_prompt(self):
        """土壤分析提示词"""
        return """你是一名土壤学专家，擅长土壤环境质量和养分分析。

专业知识要点：
- 重金属污染：单因子指数、地积累指数
- 土壤养分：氮磷钾含量评价标准
- 土壤质量：综合污染指数计算
- 修复建议：物理、化学、生物修复技术

请提供：
1. 土壤环境质量评价
2. 污染风险评估
3. 养分状况分析
4. 土地利用建议"""
    
    def _get_general_prompt(self):
        """通用环境数据提示词"""
        return """你是一名环境科学专家，擅长多种环境监测数据的分析和解读。

请根据提供的环境监测数据，进行：
1. 数据质量评估和异常值识别
2. 参数间相关性分析
3. 时间/空间变化趋势分析
4. 环境标准符合性评估
5. 专业结论和建议"""
    
    def _get_time_range(self):
        """获取时间范围信息"""
        time_columns = [col for col in self.df.columns 
                       if any(word in col.lower() for word in ['时间', 'date', 'time'])]
        if time_columns:
            time_col = time_columns[0]
            if pd.api.types.is_datetime64_any_dtype(self.df[time_col]):
                return f"{self.df[time_col].min()} 至 {self.df[time_col].max()}"
        return "未识别到明确时间信息"
    
    def _get_detailed_statistics(self):
        """获取详细统计信息"""
        stats_text = ""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:6]:  # 最多显示6个参数
            stats_text += f"{col}：均值{self.df[col].mean():.3f}±{self.df[col].std():.3f}（范围：{self.df[col].min():.3f}-{self.df[col].max():.3f}）\n"
        
        return stats_text
    
    def _generate_professional_plot(self, user_query):
        """生成专业级可视化图表"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 根据查询内容和实验类型选择图表类型
            if any(word in user_query for word in ['趋势', '变化', '时间']):
                self._plot_time_series()
            elif any(word in user_query for word in ['相关', '关系', '关联']):
                self._plot_correlation_analysis()
            elif any(word in user_query for word in ['比较', '对比', '差异']):
                self._plot_comparison()
            elif any(word in user_query for word in ['分布', '统计', '频率']):
                self._plot_distribution()
            else:
                self._plot_comprehensive_overview()
            
            plt.tight_layout()
            plot_path = f"environment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            print(f"图表生成错误：{e}")
            return None
    
    def _plot_time_series(self):
        """时间序列图"""
        time_cols = [col for col in self.df.columns if any(word in col.lower() for word in ['时间', 'date', 'time'])]
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:3]  # 取前3个数值列
        
        if time_cols and len(numeric_cols) > 0:
            time_col = time_cols[0]
            fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(12, 3*len(numeric_cols)))
            if len(numeric_cols) == 1:
                axes = [axes]
            
            for i, col in enumerate(numeric_cols):
                axes[i].plot(self.df[time_col], self.df[col], marker='o', linewidth=2, markersize=4)
                axes[i].set_ylabel(col)
                axes[i].grid(True, alpha=0.3)
                if i == len(numeric_cols)-1:
                    axes[i].set_xlabel(time_col)
            
            plt.suptitle('环境参数时间变化趋势')
    
    def _plot_correlation_analysis(self):
        """相关性分析热力图"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            corr_matrix = self.df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f')
            plt.title('环境参数相关性分析')
    
    def _plot_comparison(self):
        """多组比较图"""
         # 尝试找到分组列（如不同处理、不同点位等）
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            group_col = categorical_cols[0]
            value_col = numeric_cols[0]
            
            plt.subplot(1, 2, 1)
            sns.boxplot(data=self.df, x=group_col, y=value_col)
            plt.xticks(rotation=45)
            plt.title(f'{value_col}的箱线图比较')
            
            plt.subplot(1, 2, 2)
            group_means = self.df.groupby(group_col)[value_col].mean()
            group_means.plot(kind='bar', alpha=0.7)
            plt.title(f'{value_col}的均值比较')
            plt.xticks(rotation=45)
    
    def _plot_distribution(self):
        """分布直方图"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            n_cols = min(3, len(numeric_cols))
            fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))
            if n_cols == 1:
                axes = [axes]
            
            for i, col in enumerate(numeric_cols[:n_cols]):
                axes[i].hist(self.df[col], bins=20, alpha=0.7, edgecolor='black')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('频次')
                axes[i].set_title(f'{col}的分布')
    
    def _plot_comprehensive_overview(self):
        """综合概览图"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:4]
        
        if len(numeric_cols) > 0:
            # 创建2x2的子图布局
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:4]):
                if i < 4:
                    if i % 2 == 0:  # 左侧图：箱线图
                        sns.boxplot(y=self.df[col], ax=axes[i])
                        axes[i].set_ylabel(col)
                    else:  # 右侧图：直方图
                        axes[i].hist(self.df[col], bins=15, alpha=0.7, edgecolor='black')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('频次')

            plt.suptitle('环境参数综合概览')
#创建演示界面
def create_environment_science_interface():
    assistant = None
    
    with gr.Blocks(theme=gr.themes.Soft(), title="环境科学AI实验助手") as demo:
        gr.Markdown("""
        # 🔬 环境科学AI实验助手
        **智能分析多种环境科学实验数据 - 支持水质检测、气候变化、生态监测、土壤分析等**
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 实验数据输入")
                api_key = gr.Textbox(
                    label="智谱AI API Key",
                    placeholder="请输入您的API Key",
                    type="password"
                )
                
                data_input = gr.Radio(
                    choices=["使用示例数据", "上传CSV文件"],
                    label="数据来源",
                    value="使用示例数据"
                )
                
                example_selector = gr.Dropdown(
                    choices=[
                        "森林火险监测数据",
                        "水质检测数据(COD/六价铬)", 
                        "气候变化实验数据",
                        "生态监测数据",
                        "土壤分析数据"
                    ],
                    label="选择示例数据类型",
                    value="森林火险监测数据"
                )
                
                file_upload = gr.File(
                    label="上传实验数据文件(CSV格式)",
                    file_types=[".csv"],
                    visible=False
                )
                
                load_btn = gr.Button("加载数据", variant="primary")
                status_display = gr.Textbox(label="加载状态", lines=3, interactive=False)
            with gr.Column(scale=2):
                gr.Markdown("### 🔍 智能实验分析")
                user_query = gr.Textbox(
                    label="分析指令",
                    placeholder="例如：分析水质参数达标情况、评估气候变化趋势、比较不同处理效果...",
                    lines=3
                )
                
                with gr.Row():
                    analyze_btn = gr.Button("🔬 智能分析", variant="primary")
                    stats_btn = gr.Button("📊 数据概览")
                    suggest_btn = gr.Button("💡 分析建议")
                
                analysis_output = gr.Textbox(label="专业分析结果", lines=10)
                plot_output = gr.Image(label="分析图表", height=400)
        
        # 示例分析指令
        gr.Markdown("### 💡 专业分析示例")
        examples = gr.Examples(
            examples=[
                "分析主要环境参数的时间变化趋势",
                "评估水质指标是否符合相关环境标准", 
                "比较不同处理组间的显著性差异",
                "识别影响环境质量的关键驱动因子",
                "生成完整的实验分析报告并提出建议"
            ],
            inputs=user_query
        )
        
        # 分析历史记录
        with gr.Accordion("📋 分析历史记录", open=False):
            history_display = gr.Dataframe(
                headers=["时间", "实验类型", "分析问题", "简要结果"],
                interactive=False,
            )
        
        def toggle_file_visibility(choice):
            return gr.File(visible=(choice == "上传CSV文件"))
        
        def get_example_data(example_type):
            """获取示例数据"""
            examples = {
                "森林火险监测数据": """时间,监测点位,温度_℃,相对湿度_%,风速_m/s,死可燃物含水率_%
2024-06-01 08:00,林外100m,18.5,65.2,1.8,12.3
2024-06-01 12:00,林外100m,22.3,58.1,2.1,10.8
2024-06-01 16:00,林外100m,25.1,52.3,1.9,9.5
2024-06-01 08:00,林内100m,17.8,72.5,1.2,15.2
2024-06-01 12:00,林内100m,21.5,68.3,1.1,13.8
2024-06-01 16:00,林内100m,24.2,63.1,1.0,12.1""",
                
                "水质检测数据(COD/六价铬)": """采样点,采样时间,COD(mg/L),六价铬(mg/L),PH,氨氮(mg/L)
A点,2024-05-01,25.3,0.02,7.2,0.15
B点,2024-05-01,18.7,0.08,6.8,0.22
C点,2024-05-01,32.1,0.15,7.5,0.18
A点,2024-06-01,22.8,0.03,7.1,0.12
B点,2024-06-01,20.3,0.06,6.9,0.19
C点,2024-06-01,28.5,0.12,7.3,0.16""",
                                "气候变化实验数据": """处理组,时间,温度_℃,CO2_ppm,土壤湿度_%,生物量_g
对照组,2024-01,15.2,420,25.3,45.2
增温组,2024-01,18.5,420,24.8,48.7
对照组,2024-02,16.8,422,26.1,47.3
增温组,2024-02,20.1,422,25.2,52.1
对照组,2024-03,18.3,425,27.2,50.8
增温组,2024-03,22.6,425,26.3,55.9"""
            }
            return examples.get(example_type, examples["森林火险监测数据"])
        
        def initialize_assistant(api_key, data_choice, example_type, file):
            nonlocal assistant
            if not api_key.strip():
                return "请输入有效的API Key", None, gr.Button(interactive=False), gr.DataFrame()
            
            assistant = EnvironmentalScienceAssistant(api_key.strip())
            
            if data_choice == "使用示例数据":
                content = get_example_data(example_type)
                success, message = assistant.load_data(content, "csv")
            else:
                if file is None:
                    return "请上传数据文件", None, gr.Button(interactive=False), gr.DataFrame()
                # Gradio返回的file可能是字符串路径或文件对象
                try:
                    # 尝试作为路径字符串处理
                    if isinstance(file, str):
                        file_path = file
                    else:
                        file_path = file.name
                    
                    # 尝试多种编码方式读取CSV文件
                    content = None
                    for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if content is None:
                        return "❌ 文件编码不支持，请使用UTF-8或GBK编码", None, gr.Button(interactive=False), gr.DataFrame()
                    
                    success, message = assistant.load_data(content, "csv")
                except Exception as e:
                    return f"❌ 文件读取失败：{str(e)}", None, gr.Button(interactive=False), gr.DataFrame()
            
            if success:
                # 显示分析历史（初始为空）
                history_df = pd.DataFrame(assistant.analysis_history)
                if not history_df.empty:
                    history_display = history_df[['timestamp', 'experiment_type', 'query', 'response']]
                else:
                    history_display = pd.DataFrame(columns=["时间", "实验类型", "分析问题", "简要结果"])
                
                return message, gr.Button(interactive=True), gr.Button(interactive=True), history_display
            else:
                return message, None, gr.Button(interactive=False), gr.DataFrame()
        
        def perform_analysis(query):
            nonlocal assistant
            if assistant is None:
                return "请先初始化助手并加载数据", None, gr.DataFrame()
            
            result_text, result_plot = assistant.analyze_experiment(query)
            
            # 更新历史记录显示
            history_df = pd.DataFrame(assistant.analysis_history)
            if not history_df.empty:
                history_display = history_df[['timestamp', 'experiment_type', 'query', 'response']]
            else:
                history_display = pd.DataFrame(columns=["时间", "实验类型", "分析问题", "简要结果"])
            
            return result_text, result_plot, history_display
        
        def show_data_overview():
            nonlocal assistant
            if assistant is None or assistant.df is None:
                return "请先加载数据", None, gr.DataFrame()
            
            overview_text = f"""
            【数据概览报告】
            实验类型：{assistant.experiment_type}
            数据规模：{len(assistant.df)} 行 × {len(assistant.df.columns)} 列
            
            【数据质量检查】
            - 缺失值数量：{assistant.df.isnull().sum().sum()}
            - 重复行数量：{assistant.df.duplicated().sum()}
            - 数值型参数：{len(assistant.df.select_dtypes(include=[np.number]).columns)} 个
            - 文本型参数：{len(assistant.df.select_dtypes(include=['object']).columns)} 个
            
            【统计摘要】
            {assistant.df.describe().to_string()}
            """
            
            # 生成数据分布图
            plt.figure(figsize=(10, 6))
            numeric_cols = assistant.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                assistant.df[numeric_cols].hist(bins=15, alpha=0.7, figsize=(12, 8))
                plt.suptitle('环境参数分布直方图')
                plt.tight_layout()
                plot_path = "data_overview.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plot_path = None
            
            history_df = pd.DataFrame(assistant.analysis_history)
            if not history_df.empty:
                history_display = history_df[['timestamp', 'experiment_type', 'query', 'response']]
            else:
                             history_display = pd.DataFrame(columns=["时间", "实验类型", "分析问题", "简要结果"])
            
            return overview_text, plot_path, history_display
        
        def suggest_analysis_questions():
            nonlocal assistant
            if assistant is None:
                return "请先加载数据", None, gr.DataFrame()
            
            suggestions = {
                "水质检测": [
                    "评估各水质参数是否符合《地表水环境质量标准》",
                    "分析不同采样点水质差异及污染特征",
                    "识别主要污染因子和潜在污染源"
                ],
                "气候变化": [
                    "分析温度、CO2等参数的变化趋势",
                    "评估不同处理组间的显著性差异", 
                    "预测环境因子对生态系统的潜在影响"
                ],
                "生态监测": [
                    "分析生物多样性时空变化规律",
                    "评估环境因子对群落结构的影响",
                    "识别关键物种和生态功能群"
                ],
                "土壤分析": [
                    "评估土壤环境质量和污染风险",
                    "分析养分状况和肥力水平",
                    "提出土壤修复和改良建议"
                ],
                "通用环境数据": [
                    "分析环境参数的时空变化特征",
                    "识别参数间的相关性和驱动关系",
                    "评估环境质量状况和变化趋势"
                ]
            }
            
            suggested_questions = suggestions.get(assistant.experiment_type, suggestions["通用环境数据"])
            suggestion_text = f"💡 针对【{assistant.experiment_type}】的建议分析问题：\n\n" + "\n".join([f"{i+1}. {q}" for i, q in enumerate(suggested_questions)])
            
            history_df = pd.DataFrame(assistant.analysis_history)
            if not history_df.empty:
                history_display = history_df[['timestamp', 'experiment_type', 'query', 'response']]
            else:
                history_display = pd.DataFrame(columns=["时间", "实验类型", "分析问题", "简要结果"])
            
            return suggestion_text, None, history_display
        
        # 事件绑定
        data_input.change(toggle_file_visibility, inputs=data_input, outputs=file_upload)

        load_btn.click(
            initialize_assistant,
            inputs=[api_key, data_input, example_selector, file_upload],
            outputs=[status_display, analyze_btn, suggest_btn, history_display]
        )
        
        analyze_btn.click(
            perform_analysis,
            inputs=user_query,
            outputs=[analysis_output, plot_output, history_display]
        )
        
        stats_btn.click(
            show_data_overview,
            outputs=[analysis_output, plot_output, history_display]
        )
        
        suggest_btn.click(
            suggest_analysis_questions,
            outputs=[analysis_output, plot_output, history_display]
        )
    
    return demo

if __name__ == "__main__":
    # 启动环境科学AI实验助手
    demo = create_environment_science_interface()
    
    print("🔬 环境科学AI实验助手启动中...")
    print("🌐 访问 http://localhost:7860 使用系统")
    print("💡 支持多种环境科学实验数据类型分析")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

# models/XGBoost.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')


class YieldPredictor:
    """多作物产量预测模型 - 支持玉米和小麦（修复数据泄露 + 正确时间CV）"""

    def __init__(self, data_root='./data', crop='maize'):
        """
        初始化预测器
        crop: 'maize' 或 'wheat'
        """
        self.crop = crop
        self.data_root = data_root
        self.data_path = os.path.join(data_root, 'CN', crop)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.df = None

    def load_all_data(self):
        """加载所有数据（根据作物类型）"""
        print("=" * 50)
        print(f"开始加载{self.crop}数据...")
        print("=" * 50)

        # 1. 加载产量数据
        yield_path = os.path.join(self.data_path, f'yield_{self.crop}_CN.csv')
        self.yield_df = pd.read_csv(yield_path)

        # 使用 harvest_year 作为年份
        if 'harvest_year' in self.yield_df.columns:
            self.yield_df['year'] = pd.to_numeric(self.yield_df['harvest_year'], errors='coerce')
            print("✓ 使用 harvest_year 作为年份")
        elif 'planting_year' in self.yield_df.columns:
            self.yield_df['year'] = pd.to_numeric(self.yield_df['planting_year'], errors='coerce')
            print("✓ 使用 planting_year 作为年份")

        # 删除年份为空的记录
        before = len(self.yield_df)
        self.yield_df = self.yield_df.dropna(subset=['year'])
        self.yield_df['year'] = self.yield_df['year'].astype(int)

        print(f"✓ 产量数据: {len(self.yield_df)} 条记录")
        print(f"  年份范围: {self.yield_df['year'].min()} - {self.yield_df['year'].max()}")
        print(f"  地区数量: {self.yield_df['adm_id'].nunique()}")

        # 2. 加载位置信息
        location_path = os.path.join(self.data_path, f'location_{self.crop}_CN.csv')
        self.location_df = pd.read_csv(location_path)
        print(f"✓ 位置数据: {len(self.location_df)} 个地区")

        # 3. 加载作物日历
        calendar_path = os.path.join(self.data_path, f'crop_calendar_{self.crop}_CN.csv')
        self.calendar_df = pd.read_csv(calendar_path)
        print(f"✓ 作物日历: {len(self.calendar_df)} 条记录")

        # 4. 加载种植面积
        mask_path = os.path.join(self.data_path, f'crop_mask_{self.crop}_CN.csv')
        self.mask_df = pd.read_csv(mask_path)
        print(f"✓ 种植面积: {len(self.mask_df)} 条记录")

        # 5. 加载气象数据
        meteo_path = os.path.join(self.data_path, f'meteo_{self.crop}_CN.csv')
        if os.path.exists(meteo_path):
            self.meteo_df = pd.read_csv(meteo_path)
            print(f"✓ 气象数据: {len(self.meteo_df)} 条记录")
        else:
            self.meteo_df = None
            print("⚠️ 未找到气象数据文件")

        # 6. 加载NDVI数据
        ndvi_path = os.path.join(self.data_path, f'ndvi_{self.crop}_CN.csv')
        if os.path.exists(ndvi_path):
            self.ndvi_df = pd.read_csv(ndvi_path)
            print(f"✓ NDVI数据: {len(self.ndvi_df)} 条记录")
        else:
            self.ndvi_df = None
            print("⚠️ 未找到NDVI数据文件")

        # 7. 加载FAPAR数据
        fpar_path = os.path.join(self.data_path, f'fpar_{self.crop}_CN.csv')
        if os.path.exists(fpar_path):
            self.fpar_df = pd.read_csv(fpar_path)
            print(f"✓ FAPAR数据: {len(self.fpar_df)} 条记录")
        else:
            self.fpar_df = None
            print("⚠️ 未找到FAPAR数据文件")

        # 8. 加载土壤数据
        soil_path = os.path.join(self.data_path, f'soil_{self.crop}_CN.csv')
        if os.path.exists(soil_path):
            self.soil_df = pd.read_csv(soil_path)
            print(f"✓ 土壤数据: {len(self.soil_df)} 条记录")
        else:
            self.soil_df = None
            print("⚠️ 未找到土壤数据文件")

        # 9. 加载土壤湿度数据
        soil_moisture_path = os.path.join(self.data_path, f'soil_moisture_{self.crop}_CN.csv')
        if os.path.exists(soil_moisture_path):
            self.soil_moisture_df = pd.read_csv(soil_moisture_path)
            print(f"✓ 土壤湿度数据: {len(self.soil_moisture_df)} 条记录")
        else:
            self.soil_moisture_df = None
            print("⚠️ 未找到土壤湿度数据文件")

        print("\n数据加载完成!")

    def process_meteo_data(self):
        """处理气象数据"""
        if self.meteo_df is None:
            return None

        print("\n处理气象数据...")

        meteo = self.meteo_df.copy()
        meteo['date'] = pd.to_datetime(meteo['date'], format='%Y%m%d')
        meteo['year'] = meteo['date'].dt.year

        weather_vars = ['tmin', 'tmax', 'tavg', 'prec', 'rad', 'et0', 'vpd', 'cwb']
        existing_vars = [v for v in weather_vars if v in meteo.columns]

        agg_dict = {}
        for var in existing_vars:
            if var in ['prec', 'et0', 'cwb']:
                agg_dict[var] = 'sum'
            else:
                agg_dict[var] = 'mean'

        meteo_agg = meteo.groupby(['adm_id', 'year']).agg(agg_dict).reset_index()

        print(f"  气象数据聚合后: {len(meteo_agg)} 条记录")
        return meteo_agg

    def process_ndvi_data(self):
        """处理NDVI数据"""
        if self.ndvi_df is None:
            return None

        print("\n处理NDVI数据...")

        ndvi = self.ndvi_df.copy()
        ndvi['date'] = pd.to_datetime(ndvi['date'], format='%Y%m%d')
        ndvi['year'] = ndvi['date'].dt.year

        ndvi_agg = ndvi.groupby(['adm_id', 'year']).agg({
            'ndvi': ['mean', 'max', 'min', 'std']
        }).reset_index()

        ndvi_agg.columns = ['adm_id', 'year', 'ndvi_mean', 'ndvi_max', 'ndvi_min', 'ndvi_std']

        print(f"  NDVI数据聚合后: {len(ndvi_agg)} 条记录")
        return ndvi_agg

    def process_fpar_data(self):
        """处理FAPAR数据"""
        if self.fpar_df is None:
            return None

        print("\n处理FAPAR数据...")

        fpar = self.fpar_df.copy()
        fpar['date'] = pd.to_datetime(fpar['date'], format='%Y%m%d')
        fpar['year'] = fpar['date'].dt.year

        fpar_col = None
        for col in fpar.columns:
            if 'fpar' in col.lower():
                fpar_col = col
                break

        if fpar_col is None:
            print("  ⚠️ 未找到FAPAR数据列")
            return None

        fpar_agg = fpar.groupby(['adm_id', 'year']).agg({
            fpar_col: ['mean', 'max', 'min', 'std']
        }).reset_index()

        fpar_agg.columns = ['adm_id', 'year', 'fpar_mean', 'fpar_max', 'fpar_min', 'fpar_std']

        print(f"  FAPAR数据聚合后: {len(fpar_agg)} 条记录")
        return fpar_agg

    def process_soil_moisture_data(self):
        """处理土壤湿度数据"""
        if self.soil_moisture_df is None:
            return None

        print("\n处理土壤湿度数据...")

        soil_m = self.soil_moisture_df.copy()
        soil_m['date'] = pd.to_datetime(soil_m['date'], format='%Y%m%d')
        soil_m['year'] = soil_m['date'].dt.year

        moisture_cols = []
        for col in soil_m.columns:
            if col in ['ssm', 'rsm']:
                moisture_cols.append(col)

        if len(moisture_cols) == 0:
            print("  ⚠️ 未找到土壤湿度数据列")
            return None

        agg_dict = {col: ['mean', 'max', 'min'] for col in moisture_cols}
        soil_m_agg = soil_m.groupby(['adm_id', 'year']).agg(agg_dict).reset_index()

        flat_cols = ['adm_id', 'year']
        for col in moisture_cols:
            for stat in ['mean', 'max', 'min']:
                flat_cols.append(f'{col}_{stat}')
        soil_m_agg.columns = flat_cols

        print(f"  土壤湿度数据聚合后: {len(soil_m_agg)} 条记录")
        return soil_m_agg

    def preprocess_data(self, use_lag_features=True, add_trend_features=True):
        """
        数据预处理和特征工程
        use_lag_features: 是否使用历史产量特征（设为False可避免数据泄露）
        add_trend_features: 是否添加时间趋势和气候异常特征
        """
        print("\n" + "=" * 50)
        print("开始数据预处理...")
        print("=" * 50)

        df = self.yield_df.copy()

        # 添加位置信息
        df = df.merge(
            self.location_df[['adm_id', 'latitude', 'longitude', 'region_area']],
            on='adm_id', how='left'
        )
        print(f"✓ 添加位置信息")

        # 添加作物日历
        df = df.merge(
            self.calendar_df[['adm_id', 'sos', 'eos']],
            on='adm_id', how='left'
        )
        df['growing_days'] = df['eos'] - df['sos']
        print(f"✓ 添加作物日历")

        # 添加种植面积
        df = df.merge(
            self.mask_df[['adm_id', 'crop_area_percentage']],
            on='adm_id', how='left'
        )
        print(f"✓ 添加种植面积")

        # 添加土壤数据
        if self.soil_df is not None:
            soil_cols = [col for col in self.soil_df.columns if col not in ['adm_id', 'crop_name']]
            df = df.merge(self.soil_df[['adm_id'] + soil_cols], on='adm_id', how='left')
            print(f"✓ 添加土壤数据 ({len(soil_cols)}个特征)")

        # 添加气象数据
        meteo_agg = self.process_meteo_data()
        if meteo_agg is not None:
            df = df.merge(meteo_agg, on=['adm_id', 'year'], how='left')
            print(f"✓ 添加气象数据")

        # 添加NDVI数据
        ndvi_agg = self.process_ndvi_data()
        if ndvi_agg is not None:
            df = df.merge(ndvi_agg, on=['adm_id', 'year'], how='left')
            print(f"✓ 添加NDVI数据")

        # 添加FAPAR数据
        fpar_agg = self.process_fpar_data()
        if fpar_agg is not None:
            df = df.merge(fpar_agg, on=['adm_id', 'year'], how='left')
            print(f"✓ 添加FAPAR数据")

        # 添加土壤湿度数据
        soil_m_agg = self.process_soil_moisture_data()
        if soil_m_agg is not None:
            df = df.merge(soil_m_agg, on=['adm_id', 'year'], how='left')
            print(f"✓ 添加土壤湿度数据")

        # 特征工程
        print("\n开始特征工程...")

        df['adm_code'] = df['adm_id'].astype('category').cat.codes
        df = df.sort_values(['adm_id', 'year'])

        # ========== 新增：时间趋势特征 ==========
        if add_trend_features:
            # 全局时间趋势（捕捉品种改良、技术进步）
            df['year_normalized'] = (df['year'] - 2000) / 20
            df['year_squared'] = df['year_normalized'] ** 2

            # 地区特定的时间趋势
            df['year_region'] = df.groupby('adm_id')['year_normalized'].transform(
                lambda x: x - x.mean()
            )
            print("✓ 添加时间趋势特征（year_normalized, year_squared, year_region）")

        # ========== 新增：气候异常特征 ==========
        if add_trend_features and meteo_agg is not None:
            # 计算每个地区各气候变量的历史均值
            climate_vars = ['tmin', 'tmax', 'tavg', 'prec']
            for var in climate_vars:
                if var in df.columns:
                    # 地区历史均值（不包括当年，避免泄露）
                    var_mean = df.groupby('adm_id')[var].transform(
                        lambda x: x.expanding(min_periods=1).mean().shift(1)
                    )
                    df[f'{var}_anomaly'] = df[var] - var_mean
            print("✓ 添加气候异常特征（相对于历史均值的偏差）")

        # 历史产量特征（可选择是否使用，强烈建议设为False）
        if use_lag_features:
            df['yield_lag1'] = df.groupby('adm_id')['yield'].shift(1)
            df['yield_lag2'] = df.groupby('adm_id')['yield'].shift(2)
            df['yield_ma3'] = df.groupby('adm_id')['yield'].transform(
                lambda x: x.rolling(3, min_periods=1).mean().shift(1)
            )
            df['yield_growth'] = df.groupby('adm_id')['yield'].pct_change() * 100
            print("✓ 添加历史产量特征（lag1, lag2, ma3, growth）")
        else:
            print("⚠️ 跳过历史产量特征（避免数据泄露）")

        # 删除产量为空的记录
        before_count = len(df)
        df = df.dropna(subset=['yield'])
        after_count = len(df)
        print(f"✓ 删除缺失值: {before_count - after_count} 条")

        # 选择特征列
        base_features = ['adm_code', 'year', 'latitude', 'longitude', 'region_area',
                         'growing_days', 'crop_area_percentage']

        # 时间趋势特征
        trend_features = []
        if add_trend_features:
            trend_candidates = ['year_normalized', 'year_squared', 'year_region']
            trend_features = [f for f in trend_candidates if f in df.columns]

        # 时间序列特征（只有 use_lag_features=True 时才添加）
        if use_lag_features:
            time_features = ['yield_lag1', 'yield_lag2', 'yield_ma3', 'yield_growth']
            available_time_features = [f for f in time_features if f in df.columns]
        else:
            available_time_features = []

        weather_features = []
        if meteo_agg is not None:
            weather_candidates = ['tmin', 'tmax', 'tavg', 'prec', 'rad', 'et0', 'vpd', 'cwb']
            weather_features = [f for f in weather_candidates if f in df.columns]

        # 气候异常特征
        anomaly_features = []
        if add_trend_features:
            anomaly_candidates = [f for f in df.columns if f.endswith('_anomaly')]
            anomaly_features = [f for f in anomaly_candidates if f in df.columns]

        ndvi_features = []
        if ndvi_agg is not None:
            ndvi_candidates = ['ndvi_mean', 'ndvi_max', 'ndvi_min', 'ndvi_std']
            ndvi_features = [f for f in ndvi_candidates if f in df.columns]

        fpar_features = []
        if fpar_agg is not None:
            fpar_candidates = ['fpar_mean', 'fpar_max', 'fpar_min', 'fpar_std']
            fpar_features = [f for f in fpar_candidates if f in df.columns]

        soil_features = []
        if self.soil_df is not None:
            soil_candidates = ['awc', 'bulk_density', 'drainage_class']
            soil_features = [f for f in soil_candidates if f in df.columns]

        soil_moisture_features = []
        if soil_m_agg is not None:
            moisture_candidates = [col for col in df.columns
                                   if col.startswith('ssm_') or col.startswith('rsm_')]
            soil_moisture_features = [f for f in moisture_candidates if f in df.columns]

        self.feature_cols = (base_features + trend_features + available_time_features +
                             weather_features + anomaly_features + ndvi_features +
                             fpar_features + soil_features + soil_moisture_features)
        self.feature_cols = [f for f in self.feature_cols if f in df.columns]

        before_drop = len(df)
        df = df.dropna(subset=self.feature_cols)
        after_drop = len(df)
        print(f"✓ 删除特征缺失值: {before_drop - after_drop} 条")

        self.df = df
        print(f"\n最终特征 ({len(self.feature_cols)}个): {self.feature_cols[:10]}...")
        print(f"有效数据量: {len(self.df)} 条")
        print(f"年份范围: {self.df['year'].min()} - {self.df['year'].max()}")

        return self.df

    def train_time_series_cv(self, n_splits=5, use_lag_features=False, min_train_years=5):
        """
        使用正确的时间序列交叉验证训练模型
        n_splits: 交叉验证折数
        use_lag_features: 是否使用历史产量特征（强烈建议False）
        min_train_years: 最小训练年份数
        """
        print("\n" + "=" * 50)
        print(f"开始训练{self.crop}模型（时间序列交叉验证）...")
        print("=" * 50)

        if self.df is None:
            raise ValueError("请先调用 preprocess_data()")

        # 准备数据
        X = self.df[self.feature_cols]
        y = self.df['yield']

        # 获取所有唯一年份并排序
        unique_years = sorted(self.df['year'].unique())
        print(f"数据年份范围: {unique_years[0]} - {unique_years[-1]}")
        print(f"地区数量: {self.df['adm_id'].nunique()}")

        # 正确的滚动时间序列交叉验证
        # 每次用 [start_year, split_year] 训练，预测 split_year+1
        cv_scores = []

        # 确保有足够的年份进行交叉验证
        if len(unique_years) < min_train_years + 2:
            print(f"⚠️ 年份不足，无法进行有效的时间序列CV")
            return None, None

        # 动态确定测试年份
        # 至少用前 min_train_years 年训练，后面的年份依次作为测试
        test_years = []
        for i in range(min_train_years, len(unique_years) - 1):
            test_years.append(unique_years[i])

        # 限制折数
        n_splits = min(n_splits, len(test_years))
        if n_splits < 2:
            print(f"⚠️ 可用的测试年份不足 {n_splits} 折")
            n_splits = max(1, len(test_years))

        # 均匀选择测试年份
        if len(test_years) > n_splits:
            step = len(test_years) // n_splits
            test_years = test_years[::step][:n_splits]

        fold = 1
        for test_year in test_years:
            # 训练集：所有年份 < test_year 的数据
            train_mask = self.df['year'] < test_year
            # 测试集：仅 test_year 这一年
            test_mask = self.df['year'] == test_year

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            if len(X_train) == 0 or len(X_test) == 0:
                continue

            # 获取年份信息
            train_years = sorted(self.df[train_mask]['year'].unique())
            test_years_fold = sorted(self.df[test_mask]['year'].unique())

            print(f"\n第 {fold} 折:")
            print(f"  训练年份: {train_years[0]}-{train_years[-1]} ({len(X_train)}条, {len(train_years)}年)")
            print(f"  测试年份: {test_years_fold} ({len(X_test)}条)")

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 使用更保守的模型参数（防止过拟合）
            model = XGBRegressor(
                n_estimators=100,  # 增加树数量，配合早停
                learning_rate=0.03,  # 降低学习率
                max_depth=3,  # 更浅的树
                min_child_weight=3,  # 正则化
                subsample=0.7,  # 降低采样率
                colsample_bytree=0.7,
                reg_alpha=0.1,  # L1正则化
                reg_lambda=1.0,  # L2正则化
                random_state=42,
                verbosity=0,
                early_stopping_rounds=10  # 早停
            )

            # 训练时使用验证集（从训练集中分出20%）
            from sklearn.model_selection import train_test_split
            X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                X_train_scaled, y_train, test_size=0.2, random_state=42
            )

            model.fit(
                X_train_sub, y_train_sub,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # 评估
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            cv_scores.append({'fold': fold, 'test_year': test_year, 'r2': r2, 'mae': mae})
            print(f"  测试集 R²: {r2:.4f}, MAE: {mae:.4f} 吨/公顷")

            fold += 1

        if len(cv_scores) == 0:
            print("⚠️ 没有有效的交叉验证折")
            return None, None

        # 输出交叉验证结果
        print("\n" + "=" * 50)
        print("交叉验证结果汇总:")
        print("=" * 50)
        avg_r2 = np.mean([s['r2'] for s in cv_scores])
        std_r2 = np.std([s['r2'] for s in cv_scores])
        avg_mae = np.mean([s['mae'] for s in cv_scores])

        print(f"平均 R²: {avg_r2:.4f} (±{std_r2:.4f})")
        print(f"平均 MAE: {avg_mae:.4f} 吨/公顷")

        # 按年份显示
        print("\n各年份表现:")
        for s in cv_scores:
            print(f"  {s['test_year']}年: R²={s['r2']:.4f}, MAE={s['mae']:.4f}")

        # 使用全部数据训练最终模型
        print("\n" + "=" * 50)
        print("使用全部数据训练最终模型...")
        print("=" * 50)

        X_scaled = self.scaler.fit_transform(X)

        self.model = XGBRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=3,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        )

        self.model.fit(X_scaled, y)

        # 最终模型评估（在训练集上，仅供参考）
        y_pred = self.model.predict(X_scaled)
        final_r2 = r2_score(y, y_pred)
        final_mae = mean_absolute_error(y, y_pred)

        print(f"最终模型（全量数据）:")
        print(f"  R²: {final_r2:.4f}")
        print(f"  MAE: {final_mae:.4f} 吨/公顷")

        # 检查泛化能力
        if final_r2 - avg_r2 > 0.3:
            print(f"\n⚠️ 警告：可能存在过拟合！")
            print(f"   全量R²({final_r2:.3f}) - CV平均R²({avg_r2:.3f}) = {final_r2 - avg_r2:.3f}")

        return avg_r2, avg_mae

    def train(self, test_year=None, use_lag_features=False):
        """
        训练模型（简单训练/测试分割）
        test_year: 测试集起始年份（用该年份及之后作为测试）
        use_lag_features: 是否使用历史产量特征
        """
        print("\n" + "=" * 50)
        print(f"开始训练{self.crop}模型...")
        print("=" * 50)

        if self.df is None:
            raise ValueError("请先调用 preprocess_data()")

        # 获取所有年份
        years = sorted(self.df['year'].unique())

        # 使用最后3年作为测试集（避免数据泄露）
        if test_year is None:
            test_year = years[-3]  # 最后3年作为测试
            print(f"使用最后3年作为测试集: {test_year}年及之后")

        X = self.df[self.feature_cols]
        y = self.df['yield']

        # 重要：确保训练集和测试集没有时间重叠
        train_mask = self.df['year'] < test_year  # 严格小于，不包含
        test_mask = self.df['year'] >= test_year

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        train_years = self.df[train_mask]['year'].unique()
        test_years = self.df[test_mask]['year'].unique()

        print(f"训练集: {len(X_train)} 条 ({train_years[0]}-{train_years[-1]})")
        print(f"测试集: {len(X_test)} 条 ({test_years[0]}-{test_years[-1]})")

        if len(X_train) == 0:
            print("错误：训练集为空！")
            return None, None

        # 标准化（只在训练集上fit）
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test) if len(X_test) > 0 else X_test

        # 使用更保守的模型参数（避免过拟合）
        self.model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=3,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        )

        self.model.fit(X_train_scaled, y_train)

        # 评估
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled) if len(X_test) > 0 else []

        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)

        print(f"\n训练集表现:")
        print(f"  MAE: {train_mae:.3f} 吨/公顷")
        print(f"  R²:  {train_r2:.3f}")

        if len(X_test) > 0:
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)
            print(f"\n测试集表现:")
            print(f"  MAE: {test_mae:.3f} 吨/公顷")
            print(f"  R²:  {test_r2:.3f}")

            # 检查过拟合
            if train_r2 - test_r2 > 0.15:
                print(f"\n⚠️ 警告：可能存在过拟合！")
                print(f"   训练R²({train_r2:.3f}) - 测试R²({test_r2:.3f}) = {train_r2 - test_r2:.3f}")

            return test_mae, test_r2
        else:
            print(f"\n⚠️ 无测试数据")
            return None, None

    def predict(self, adm_id, year):
        """预测产量"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train()")

        history = self.df[self.df['adm_id'] == adm_id].sort_values('year')

        if len(history) == 0:
            return {'success': False, 'error': f'未找到地区 {adm_id}'}

        latest = history.iloc[-1]

        if len(history) >= 3:
            trend = (history['yield'].iloc[-1] - history['yield'].iloc[-3]) / 2
        else:
            trend = 0

        predicted_yield = latest['yield'] + trend * (year - latest['year'])

        return {
            'success': True,
            'adm_id': adm_id,
            'crop': self.crop,
            'year': year,
            'predicted_yield': round(predicted_yield, 2),
            'unit': '吨/公顷',
            'base_year': int(latest['year']),
            'base_yield': round(latest['yield'], 2)
        }

    def save_model(self, path=None):
        """保存模型"""
        if path is None:
            path = f'./models/saved_models/{self.crop}_predictor.pkl'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'crop': self.crop
        }, path)
        print(f"\n✓ {self.crop}模型已保存到: {path}")

    def load_model(self, path=None):
        """加载模型"""
        if path is None:
            path = f'./models/saved_models/{self.crop}_predictor.pkl'
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_cols = data['feature_cols']
            print(f"✓ {self.crop}模型已加载: {path}")
            return True
        else:
            print(f"⚠️ 模型文件不存在: {path}")
            return False


# ========== 主程序入口 ==========
if __name__ == "__main__":
    import sys
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"工作目录: {os.getcwd()}")

    # 训练玉米模型（不使用历史产量特征，避免数据泄露）
    print("\n" + "=" * 60)
    print("训练玉米产量预测模型（使用时间序列交叉验证）")
    print("=" * 60)
    maize_predictor = YieldPredictor(data_root='./data', crop='maize')
    maize_predictor.load_all_data()
    maize_predictor.preprocess_data(use_lag_features=False, add_trend_features=True)  # 添加趋势特征
    maize_predictor.train_time_series_cv(n_splits=5)  # 使用修复后的时间序列交叉验证
    maize_predictor.save_model()

    # 训练小麦模型
    print("\n" + "=" * 60)
    print("训练小麦产量预测模型（使用时间序列交叉验证）")
    print("=" * 60)
    wheat_predictor = YieldPredictor(data_root='./data', crop='wheat')
    wheat_predictor.load_all_data()
    wheat_predictor.preprocess_data(use_lag_features=False, add_trend_features=True)
    wheat_predictor.train_time_series_cv(n_splits=5)
    wheat_predictor.save_model()

    print("\n✅ 所有模型训练完成！")
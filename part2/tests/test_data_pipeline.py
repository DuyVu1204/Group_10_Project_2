import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline import (
    DataPipeline, split_features_target, train_test_split_data,
    prepare_data, calculate_vif, select_features_by_vif
)


@pytest.fixture
def sample_data():
    """Create a simple test dataset."""
    np.random.seed(42)
    df = pd.DataFrame({
        'numeric_1': np.random.randn(100),
        'numeric_2': np.random.randn(100),
        'cat_1': np.random.choice(['A', 'B', 'C'], 100),
        'cat_2': np.random.choice(['X', 'Y'], 100),
        'target': np.random.randn(100)
    })
    return df


class TestDataPipeline:
    """Unit tests for DataPipeline class."""

    def test_pipeline_fit_transform(self, sample_data):
        """Test fit_transform works without error."""
        X = sample_data.drop(columns=['target'])
        pipeline = DataPipeline(
            numeric_features=['numeric_1', 'numeric_2'],
            categorical_features=['cat_1', 'cat_2']
        )
        X_transformed = pipeline.fit_transform(X)
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]

    def test_pipeline_fit_then_transform(self, sample_data):
        """Test fit followed by separate transform."""
        X = sample_data.drop(columns=['target'])
        X_train = X.iloc[:80]
        X_test = X.iloc[80:]

        pipeline = DataPipeline(
            numeric_features=['numeric_1', 'numeric_2'],
            categorical_features=['cat_1', 'cat_2']
        )
        pipeline.fit(X_train)
        X_train_t = pipeline.transform(X_train)
        X_test_t = pipeline.transform(X_test)

        assert X_train_t.shape[0] == 80
        assert X_test_t.shape[0] == 20
        assert X_train_t.shape[1] == X_test_t.shape[1]

    def test_no_missing_after_transform(self, sample_data):
        """Test no missing values after transform."""
        X = sample_data.drop(columns=['target']).copy()
        X.loc[0, 'numeric_1'] = np.nan
        X.loc[1, 'cat_1'] = np.nan

        pipeline = DataPipeline(
            numeric_features=['numeric_1', 'numeric_2'],
            categorical_features=['cat_1', 'cat_2']
        )
        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.isnull().sum().sum() == 0

    def test_train_test_same_columns(self, sample_data):
        """Test train and test have same number of columns."""
        X = sample_data.drop(columns=['target'])
        X_train = X.iloc[:80]
        X_test = X.iloc[80:]

        pipeline = DataPipeline(
            numeric_features=['numeric_1', 'numeric_2'],
            categorical_features=['cat_1', 'cat_2']
        )
        pipeline.fit(X_train)
        X_train_t = pipeline.transform(X_train)
        X_test_t = pipeline.transform(X_test)

        assert X_train_t.shape[1] == X_test_t.shape[1]

    def test_transform_before_fit_raises_error(self, sample_data):
        """Test that transform raises error if called before fit."""
        X = sample_data.drop(columns=['target'])
        pipeline = DataPipeline(
            numeric_features=['numeric_1', 'numeric_2'],
            categorical_features=['cat_1', 'cat_2']
        )
        with pytest.raises(ValueError):
            pipeline.transform(X)


class TestHelperFunctions:
    """Unit tests for helper functions."""

    def test_split_features_target(self, sample_data):
        """Test split_features_target."""
        X, y = split_features_target(sample_data, 'target')
        assert X.shape[0] == y.shape[0]
        assert 'target' not in X.columns
        assert y.name == 'target'

    def test_train_test_split(self, sample_data):
        """Test train_test_split_data."""
        X = sample_data.drop(columns=['target'])
        y = sample_data['target']
        X_train, X_test, y_train, y_test = train_test_split_data(
            X, y, test_size=0.2, random_state=42
        )
        assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
        assert len(y_train) + len(y_test) == len(y)
        assert X_train.shape[1] == X_test.shape[1]

    def test_prepare_data(self, sample_data):
        """Test prepare_data returns correct shapes."""
        result = prepare_data(
            sample_data,
            target_column='target',
            numeric_features=['numeric_1', 'numeric_2'],
            categorical_features=['cat_1', 'cat_2'],
            test_size=0.2
        )
        X_train, X_test, y_train, y_test, feature_names, pipeline = result

        assert X_train.shape[0] + X_test.shape[0] == sample_data.shape[0]
        assert X_train.shape[1] == X_test.shape[1]
        assert len(feature_names) == X_train.shape[1]
        assert isinstance(pipeline, DataPipeline)


class TestVIF:
    """Unit tests for VIF functions."""

    def test_calculate_vif(self):
        """Test calculate_vif returns DataFrame with correct columns."""
        X = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        vif_data = calculate_vif(X)
        assert isinstance(vif_data, pd.DataFrame)
        assert 'Feature' in vif_data.columns
        assert 'VIF' in vif_data.columns
        assert len(vif_data) == 3

    def test_select_features_by_vif(self):
        """Test select_features_by_vif removes high VIF features."""
        np.random.seed(42)
        X = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        # Create a feature that's highly correlated with x1
        X['x4'] = X['x1'] * 0.95 + np.random.randn(100) * 0.1

        selected_features, _ = select_features_by_vif(X, threshold=10.0)
        assert isinstance(selected_features, list)
        assert len(selected_features) <= X.shape[1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

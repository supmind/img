import os
import pandas as pd
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import NestedField, StringType, ListType, FloatType

class IcebergManager:
    """
    一个用于管理特征向量在Iceberg中存储的封装类。
    """
    def __init__(self, catalog_name: str, table_identifier: str):
        """
        初始化Iceberg管理器。

        它会加载指定的Catalog，并检查目标表是否存在。如果表不存在，
        它会根据预定义的Schema自动创建它。

        :param catalog_name: 要使用的Catalog的名称 (例如 'default').
        :param table_identifier: Iceberg表的完全限定名 (例如 'feature_db.vectors').
        """
        print("正在初始化Iceberg管理器...")
        self.catalog = load_catalog(catalog_name)
        self.table_identifier = table_identifier
        self.table = None

        # 定义表的Schema
        self.schema = Schema(
            NestedField("id", StringType(), required=True, doc="图像的唯一ID，通常是文件名（不含扩展名）"),
            NestedField("vec", ListType(element_id=2, element_type=FloatType(), element_required=True), doc="从图像中提取的特征向量")
        )

        self._initialize_table()

    def _initialize_table(self):
        """检查表是否存在，如果不存在则创建它。"""
        try:
            self.table = self.catalog.load_table(self.table_identifier)
            print(f"成功加载已存在的Iceberg表: {self.table_identifier}")
        except Exception:
            print(f"表 {self.table_identifier} 不存在。正在根据Schema创建新表...")
            # 为了简化，我们假设数据库/命名空间已经存在
            # 在生产环境中，您可能需要先检查并创建命名空间
            try:
                self.table = self.catalog.create_table(identifier=self.table_identifier, schema=self.schema)
                print(f"成功创建新表: {self.table_identifier}")
            except Exception as e:
                print(f"创建表时发生严重错误: {e}")
                raise e

    def append(self, data: list[dict]):
        """
        将一批新的数据追加到Iceberg表中。

        :param data: 一个字典列表，每个字典应包含 'id' 和 'vec' 键。
                     例如: [{'id': 'img1', 'vec': [0.1, 0.2, ...]}, ...]
        """
        if not data:
            print("要追加的数据为空，已跳过。")
            return

        if not self.table:
            print("错误: Iceberg表未初始化，无法追加数据。")
            return

        try:
            df = pd.DataFrame(data)
            # 使用Iceberg表的append方法
            self.table.append(df)
            print(f"成功向 {self.table_identifier} 追加了 {len(df)} 条记录。")
        except Exception as e:
            print(f"向Iceberg表追加数据时发生错误: {e}")
            # 在实际应用中，这里可能需要更复杂的错误处理和重试逻辑
            raise e

# --- 使用示例 ---
if __name__ == '__main__':
    # 这是一个如何使用IcebergManager的示例。
    # 在运行此示例之前，请确保已设置好以下环境变量来连接到您的REST Catalog:
    # export PYICEBERG_CATALOG__DEFAULT__URI=http://localhost:8181
    # export PYICEBERG_CATALOG__DEFAULT__S3_ENDPOINT=http://localhost:9000
    # export PYICEBERG_CATALOG__DEFAULT__S3_ACCESS_KEY_ID=minio
    # export PYICEBERG_CATALOG__DEFAULT__S3_SECRET_ACCESS_KEY=minio123

    # 并且，请确保名为 'feature_db' 的namespace/database已经在您的Catalog中创建。
    # 例如，使用 `pyspark` 或 `spark-sql` 执行:
    # CREATE NAMESPACE feature_db;

    print("--- Iceberg管理器使用示例 ---")
    try:
        # 1. 初始化管理器
        # 在实际应用中，这些参数可能来自配置文件或环境变量
        manager = IcebergManager(
            catalog_name='default',
            table_identifier='feature_db.vectors'
        )

        # 2. 准备一些虚拟数据
        dummy_data = [
            {'id': 'test_image_001', 'vec': [random.uniform(-1, 1) for _ in range(256)]},
            {'id': 'test_image_002', 'vec': [random.uniform(-1, 1) for _ in range(256)]}
        ]
        import random
        print(f"\n准备了 {len(dummy_data)} 条虚拟数据进行追加。")

        # 3. 追加数据
        manager.append(dummy_data)

        # 4. 验证数据是否已写入
        print("\n正在读取表中的总行数以进行验证...")
        df_read = manager.table.scan().to_pandas()
        print(f"验证成功！表 {manager.table_identifier} 中现在共有 {len(df_read)} 行。")
        print("最新的几行数据:")
        print(df_read.tail(2))

    except Exception as e:
        print(f"\n示例执行失败: {e}")
        print("请检查您的Iceberg REST Catalog和S3服务是否正在运行，并且环境变量是否已正确设置。")

    print("\n--- 示例执行完毕 ---")

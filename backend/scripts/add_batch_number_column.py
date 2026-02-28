#!/usr/bin/env python3
"""
添加batch_number列到clips表
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.core.database import engine
from sqlalchemy import text

def add_batch_number_column():
    """添加batch_number列到clips表"""
    print("开始添加batch_number列到clips表...")
    
    try:
        with engine.connect() as conn:
            # 检查clips表是否存在
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='clips';"))
            if not result.fetchone():
                print("❌ clips表不存在")
                return False
            
            # 检查batch_number列是否已存在
            result = conn.execute(text("PRAGMA table_info(clips);"))
            columns = [row[1] for row in result]
            
            if 'batch_number' in columns:
                print("✅ batch_number列已存在")
                return True
            
            # 添加batch_number列
            conn.execute(text("ALTER TABLE clips ADD COLUMN batch_number TEXT;"))
            conn.commit()
            print("✅ 成功添加batch_number列到clips表")
            return True
            
    except Exception as e:
        print(f"❌ 添加batch_number列失败: {e}")
        return False

def main():
    """主函数"""
    if add_batch_number_column():
        print("🎉 任务完成！")
    else:
        print("❌ 任务失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()

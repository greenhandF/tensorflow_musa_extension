import re
from collections import Counter

def analyze_log_file(filename="rank_log.log"):
    stats = Counter()
    line_count = 0
    
    try:
        print(f"🔍 正在解析日志文件: {filename} ...")
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_count += 1
                
                if line_count <= 3:
                    print(f"  [L{line_count}] {line.strip()[:80]}...")

                match_eager = re.search(r"Executing op (?P<op>\w+) in device .*device:(?P<dev>\w+):\d", line)
                
                match_graph = re.search(r"\((?P<op>\w+)\): .*device:(?P<dev>\w+):\d", line)
                
                if match_eager:
                    op_type = match_eager.group("op")
                    device = match_eager.group("dev")
                    stats[(op_type, device)] += 1
                elif match_graph:
                    op_type = match_graph.group("op")
                    device = match_graph.group("dev")
                    stats[(op_type, device)] += 1
        
        # 结果判断
        if not stats:
            print(f"\n❌ 未在日志中发现算子执行记录！")
            print(f"   已扫描行数: {line_count}")
            print("   请检查上方打印的前3行内容，确认是否包含 'Executing op' 或 'device:MUSA'。")
            return

        # 打印报表
        print("\n" + "📊 算子分布统计报告".center(60))
        print("-" * 65)
        print(f"{'算子名称 (Op Type)':<30} | {'设备 (Device)':<10} | {'出现次数':<5}")
        print("-" * 65)

        musa_total = 0
        cpu_total = 0
        
        for (op, dev), count in stats.most_common():
            print(f"{op:<30} | {dev:<10} | {count:<5}")
            if "MUSA" in dev.upper():
                musa_total += count
            else:
                cpu_total += count

        print("-" * 65)
        total = musa_total + cpu_total
        print(f"✅ MUSA 算子总计: {musa_total}")
        print(f"❌ CPU  算子总计: {cpu_total}")
        
        if total > 0:
            print(f"🚀 MUSA 覆盖率: {musa_total/total:.2%}")
        else:
            print(f"🚀 MUSA 覆盖率: N/A")
            
        if musa_total == 0 and cpu_total > 0:
            print("\n⚠️  注意: 所有算子都跑在 CPU 上。")
            print("    请确认 CDNet.py 中是否已成功加载 libmusa_plugin.so")
        
    except FileNotFoundError:
        print(f"❌ 找不到日志文件: {filename}")

if __name__ == "__main__":
    #analyze_log_file("embedding.log")
    #analyze_log_file("lr.log")
    analyze_log_file("mlp.log")

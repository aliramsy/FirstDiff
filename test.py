count = 0

with open("14.txt", "r") as f:
    for line in f:
        if line.startswith("False Positive"):
            parts = line.split("|")

            cond_score = None
            epsilon = None

            for p in parts:
                if "total_cond_scores" in p:
                    cond_score = float(p.split(":")[1].strip())
                if "epsilon" in p:
                    epsilon = float(p.split(":")[1].strip())

            if cond_score is not None and epsilon is not None:
                if cond_score > 3.5 and epsilon > 30.5:
                    count += 1

print("TPs with total_cond_scores > 1 and epsilon > 26:", count)

---
layout: leetcode-entry
title: "2462. Total Cost to Hire K Workers"
permalink: "/leetcode/problem/2023-06-26-2462-total-cost-to-hire-k-workers/"
leetcode_ui: true
entry_slug: "2023-06-26-2462-total-cost-to-hire-k-workers"
---

[2462. Total Cost to Hire K Workers](https://leetcode.com/problems/total-cost-to-hire-k-workers/description/) medium
[blog post](https://leetcode.com/problems/total-cost-to-hire-k-workers/solutions/3683531/kotlin-two-pointer-priority-queue/)
[substack](https://dmitriisamoilenko.substack.com/p/26062023-2462-total-cost-to-hire?sd=pf)
![image.png](/assets/leetcode_daily_images/85ddb762.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/257
#### Problem TLDR
The sum of the smallest cost from suffix and prefix of a `costs` size of `candidates` in `k` iterations
#### Intuition
Description of the problem is rather ambiguous: we actually need to consider `candidates` count of items from the head and from the tail of the `costs` array. Then we can use `PriorityQueue` to choose the minimum and adjust two pointers `lo` and `hi`.

#### Approach
* use separate condition, when `2 * candidates >= costs.size`
* careful with indexes, check yourself by doing dry run
* we can use separate variable `takenL` and `takenR` or just use queue's sizes to minify the code

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

        fun totalCost(costs: IntArray, k: Int, candidates: Int): Long {
            val pqL = PriorityQueue<Int>()
            val pqR = PriorityQueue<Int>()
            var lo = 0
            var hi = costs.lastIndex
            var sum = 0L
            var count = 0
            if (2 * candidates >= costs.size) while (lo <= hi) pqL.add(costs[lo++])
            while (pqL.size < candidates && lo <= hi) pqL.add(costs[lo++])
            while (pqR.size < candidates && lo < hi) pqR.add(costs[hi--])
            while (lo <= hi && count++ < k) {
                if (pqR.peek() < pqL.peek()) {
                    sum += pqR.poll()
                    pqR.add(costs[hi--])
                } else {
                    sum += pqL.poll()
                    pqL.add(costs[lo++])
                }
            }
            while (pqR.isNotEmpty()) pqL.add(pqR.poll())
            while (count++ < k && pqL.isNotEmpty()) sum += pqL.poll()
            return sum
        }

```


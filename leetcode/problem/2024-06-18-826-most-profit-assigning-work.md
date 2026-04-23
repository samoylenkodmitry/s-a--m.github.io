---
layout: leetcode-entry
title: "826. Most Profit Assigning Work"
permalink: "/leetcode/problem/2024-06-18-826-most-profit-assigning-work/"
leetcode_ui: true
entry_slug: "2024-06-18-826-most-profit-assigning-work"
---

[826. Most Profit Assigning Work](https://leetcode.com/problems/most-profit-assigning-work/description/) medium
[blog post](https://leetcode.com/problems/most-profit-assigning-work/solutions/5330161/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18062024-826-most-profit-assigning?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/aYsi3nakmNk)
![2024-06-18_07-15_1.webp](/assets/leetcode_daily_images/274d7d04.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/643

#### Problem TLDR

Max profit by assigning `[profit, difficulty]` to workers any times #medium #sorting #greedy

#### Intuition

Let's start with sorting `worker` and `difficulty`.

The greedy algorithm:
* take least able worker
* take all jobs that he able to work with
* choose maximum profit job

```j
    //  2  4  6  8 10       4 5 6 7
    // 10 20 30 40 50       a b c d
    //  a  a
    //  b  b
    //  c  c  c
    //  d  d  d

    // 68 35 52 47 86          92 10 85 84 82
    // 67 17  1 81  3

    // 35 47 52 68 86          10 82 84 85 92
    // 17 81  1 67  3              d
    //  i              max = 17
    //     i           max = 81
    //        i        max = 81
    //          i      68 < 82, max = 81, use 81
    //                               d = 84, use 81
    //                                  d = 85, use 81
    //                                     d = 92
    //              i  max = 81            use 81
```

#### Approach

* pay attention that we can reuse jobs, otherwise we would have to use the PriorityQueue and `poll` each taken job

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun maxProfitAssignment(difficulty: IntArray, profit: IntArray, worker: IntArray): Int {
        val inds = profit.indices.sortedBy { difficulty[it] }
        var maxProfit = 0
        var i = 0
        return worker.sorted().sumBy { d ->
            while (i < inds.size && difficulty[inds[i]] <= d)
                maxProfit = max(maxProfit, profit[inds[i++]])
            maxProfit
        }
    }

```
```rust

    pub fn max_profit_assignment(difficulty: Vec<i32>, profit: Vec<i32>, mut worker: Vec<i32>) -> i32 {
        let (mut i, mut res, mut max, mut inds) = (0, 0, 0, (0..profit.len()).collect::<Vec<_>>());
        worker.sort_unstable(); inds.sort_unstable_by_key(|&i| difficulty[i]);
        for d in worker {
            while i < inds.len() && difficulty[inds[i]] <= d { max = max.max(profit[inds[i]]); i += 1 }
            res += max
        }; res
    }

```


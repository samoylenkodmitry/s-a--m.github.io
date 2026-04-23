---
layout: leetcode-entry
title: "3562. Maximum Profit from Trading Stocks with Discounts"
permalink: "/leetcode/problem/2025-12-16-3562-maximum-profit-from-trading-stocks-with-discounts/"
leetcode_ui: true
entry_slug: "2025-12-16-3562-maximum-profit-from-trading-stocks-with-discounts"
---

[3562. Maximum Profit from Trading Stocks with Discounts](https://leetcode.com/problems/maximum-profit-from-trading-stocks-with-discounts/description/) hard
[blog post](https://leetcode.com/problems/maximum-profit-from-trading-stocks-with-discounts/solutions/7417541/kotlin-by-samoylenkodmitry-dix2/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16122025-3562-maximum-profit-from?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UVSDgwE7IgQ)

![13ac0576-627a-49fc-a4f0-8b7776c87171 (1).webp](/assets/leetcode_daily_images/10e33abd.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1206

#### Problem TLDR

Max profit in a tree #hard

#### Intuition

```j
    // can't understand the description
    // is the 'stock' a single or many?
    // where are the stocks?
    // why example 1 can buy by both
    // and example 2 can not?
    //
    // why example two:
    // boss: buy at 3 sell at 5: 5-3=2
    // employee: buy at (4/2) sell at 8: 8-2=6
    // profit: 6+2=8, budget (2+3) - ok, its out of budget=4
    // ok, have some understanding: example 2 is out of budget
    //
    // so we have to pick either buy as a boss
    //                        or buy as a subordinate
    //
    // it can be: dp + graph DFS
    //
    // ok but how to spread budget between children?
    //
    // interesting that the prices are so low 1..50
    //
    // 30 minute, look for hints; the main difficulty is how to pick the best children and how many?
    // still the question: how to account to budget?
    //
    // just picking a single child is not working
    //
    // put children in a PriorityQueue by profit and update the tree?
    //
    // ok we have only 160 nodes; it is 2^160 ways of take/not take
    //
    // ok i kind of solved it, TLE is from kotlin slowness and usage of a HashMap
```

#### Approach

#### Complexity

- Time complexity:
$$O(n^2b^2)$$

- Space complexity:
$$O(n^2b^2)$$

#### Code

```kotlin
// 2822ms
    fun maxProfit(n: Int, p: IntArray, f: IntArray, h: Array<IntArray>, b: Int): Int {
        val ch = Array(n) { ArrayList<Int>() }; for ((p,c) in h) ch[p-1] += c-1
        val dp = HashMap<Int, Int>(); val d = HashMap<Int, Int>()
        fun dfs(i: Int, half: Int, budget: Int): Int =
            dp.getOrPut(i * 10000000 + half*10000 + budget) {
                fun maxp(j: Int, hf: Int, bgt: Int): Int = if (j == ch[i].size) 0 else
                    d.getOrPut(i * 100000000 + j * 100000 + hf*1000 + bgt) {
                        (0..bgt).maxOf { takeB -> dfs(ch[i][j], hf, takeB) + maxp(j+1, hf, bgt-takeB) }
                    }
                val p1 = maxp(0, 0, budget); val spend = if (half==0) p[i] else p[i]/2
                if (spend > budget) p1 else max(f[i] - spend + maxp(0, 1, budget-spend), p1)
            }
        return dfs(0, 0, b)
    }
```


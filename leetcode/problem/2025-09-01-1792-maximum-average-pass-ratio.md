---
layout: leetcode-entry
title: "1792. Maximum Average Pass Ratio"
permalink: "/leetcode/problem/2025-09-01-1792-maximum-average-pass-ratio/"
leetcode_ui: true
entry_slug: "2025-09-01-1792-maximum-average-pass-ratio"
---

[1792. Maximum Average Pass Ratio](https://leetcode.com/problems/maximum-average-pass-ratio/description) medium
[blog post](https://leetcode.com/problems/maximum-average-pass-ratio/solutions/7143969/kotlin-by-samoylenkodmitry-k4id/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01092025-1792-maximum-average-pass?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/STUaNN6HzxA)

![1.webp](/assets/leetcode_daily_images/84db8d16.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1099

#### Problem TLDR

Max avg ratio after assigning extra to ratios #medium

#### Intuition

Greedy works: for each extra choose the class that will make the most difference.
```j
    //[1,2],[3,5],[2,2]]
    // 2/3 vs 3/5
    // 10/15  9/15
    //
    // 2/3 4/6 2/2   vs   3/4 3/5 2/2
    // a/b c/d
    // a+1/b+1 +c/d   vs   a/b + c+1/d+1
```

#### Approach

* we don't have to validate in the end; choose only available numbers
* there is a bitmask optimization
* we can prioritize rows, cols or subs with more numbers filled

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 332ms
    fun maxAverageRatio(cls: Array<IntArray>, ext: Int): Double {
        val q = PriorityQueue<IntArray>(compareBy { (a,b) -> 1.0*a/b-1.0*(a+1)/(b+1)})
        q += cls; for (e in 1..ext) q += q.poll().also { ++it[0]; ++it[1] }
        return cls.sumOf { (a,b) -> 1.0*a/b } / cls.size
    }

```


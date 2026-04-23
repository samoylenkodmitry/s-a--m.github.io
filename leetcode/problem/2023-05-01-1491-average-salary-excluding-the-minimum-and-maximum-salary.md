---
layout: leetcode-entry
title: "1491. Average Salary Excluding the Minimum and Maximum Salary"
permalink: "/leetcode/problem/2023-05-01-1491-average-salary-excluding-the-minimum-and-maximum-salary/"
leetcode_ui: true
entry_slug: "2023-05-01-1491-average-salary-excluding-the-minimum-and-maximum-salary"
---

[1491. Average Salary Excluding the Minimum and Maximum Salary](https://leetcode.com/problems/average-salary-excluding-the-minimum-and-maximum-salary/description/) easy

```kotlin

fun average(salary: IntArray): Double = with (salary) {
    (sum() - max()!! - min()!!) / (size - 2).toDouble()
}

```

or

```

fun average(salary: IntArray): Double = salary.sorted().drop(1).dropLast(1).average()

```

[blog post](https://leetcode.com/problems/average-salary-excluding-the-minimum-and-maximum-salary/solutions/3471763/kotlin-sum-max-min/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-1052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/198
#### Intuition
Just do what is asked.

#### Approach
We can do `.fold` and iterate only once, but `sum`, `max` and `min` operators are less verbose.
We also can sort it, that will make code even shorter.
#### Complexity
- Time complexity:
$$O(n)$$, $$O(nlog(n))$$ for sorted
- Space complexity:
$$O(1)$$


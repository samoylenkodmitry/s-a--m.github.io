---
layout: leetcode-entry
title: "2698. Find the Punishment Number of an Integer"
permalink: "/leetcode/problem/2025-02-15-2698-find-the-punishment-number-of-an-integer/"
leetcode_ui: true
entry_slug: "2025-02-15-2698-find-the-punishment-number-of-an-integer"
---

[2698. Find the Punishment Number of an Integer](https://leetcode.com/problems/find-the-punishment-number-of-an-integer/description/) medium
[blog post](https://leetcode.com/problems/find-the-punishment-number-of-an-integer/solutions/6424756/kotlin-rust-by-samoylenkodmitry-noqr/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15022025-2698-find-the-punishment?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/XHxubsKDhEU)
![1.webp](/assets/leetcode_daily_images/4790f413.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/896

#### Problem TLDR

Sum x^2 if it's string partition sum = x #medium #backtracking

#### Intuition

Attention: partition can be split into any number of parts.
As the max is 1000000, the brute-force is accepted.

#### Approach

* we can skip string conversions using division by 10, 100, 1000
* we can do adding to sum or subtraction from the target; addition is more tricky, corner case is 1000
* the result set is small

#### Complexity

- Time complexity:
$$O(n*lg(n)^{2lg(n)})$$, where lg(n) is the backtracking depth, at most 6

- Space complexity:
$$O(lg(n))$$

#### Code

```kotlin

    fun punishmentNumber(n: Int) = (1..n).sumOf { x ->
        fun dfs(n: Int, s: Int): Boolean = s + n == x ||
            n > 0 && setOf(10, 100, 1000).any { dfs(n / it, s + n % it) }
        if (dfs(x * x, 0)) x * x else 0
    }

```
```rust

    pub fn punishment_number(n: i32) -> i32 {
        (1..=n).map(|x| {
            fn dfs(n: i32, t: i32) -> bool {
                n == t || n > 0 && [10, 100, 1000].iter().any(|i| dfs(n / i, t - n % i)) }
            if dfs(x * x, x) { x * x } else { 0 }
        }).sum()
    }

```
```c++

    int punishmentNumber(int n) {
        for (int s = 0; int x: {1, 9, 10, 36, 45, 55, 82, 91, 99, 100, 235, 297, 369, 370, 379, 414, 657, 675, 703, 756, 792, 909, 918, 945, 964, 990, 991, 999, 1000, 1001})
            if (x > n) return s; else s += x * x; return 0;
    }

```


---
layout: leetcode-entry
title: "2409. Count Days Spent Together"
permalink: "/leetcode/problem/2026-10-24-2409-count-days-spent-together/"
leetcode_ui: true
entry_slug: "2026-10-24-2409-count-days-spent-together"
---

[2409. Count Days Spent Together](https://leetcode.com/problems/count-days-spent-together/solutions/8586256/kotlin-rust-by-samoylenkodmitry-63yz/) easy
[substack](https://dmitriisamoilenko.substack.com/p/24102026-2409-count-days-spent-together?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/vDeyO8Bamc4)

https://dmitrysamoylenko.com/leetcode/

![24.10.2026.webp](/assets/leetcode_daily_images/24.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1521

#### Problem TLDR

Days range intersection

#### Intuition

Convert to day of a year

#### Approach

* we can compare strings directly

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun countDaysTogether(aa: String, la: String, ab: String, lb: String): Int {
        val days = listOf(0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        fun d(s: String) = days.take(s.take(2).toInt()).sum() + s.drop(3).toInt()
        return max(0, d(minOf(la, lb)) - d(maxOf(aa, ab)) + 1)
    }
```
```rust
    pub fn count_days_together(aa: String, la: String, ab: String, lb: String) -> i32 {
        let d = |s: &str| [0,31,59,90,120,151,181,212,243,273,304,334][s[..2].parse::<usize>().unwrap() - 1] + s[3..].parse::<i32>().unwrap();
        (d(&la.min(lb)) - d(&aa.max(ab)) + 1).max(0)
    }
```


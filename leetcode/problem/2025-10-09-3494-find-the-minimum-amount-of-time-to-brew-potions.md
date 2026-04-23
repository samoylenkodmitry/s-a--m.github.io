---
layout: leetcode-entry
title: "3494. Find the Minimum Amount of Time to Brew Potions"
permalink: "/leetcode/problem/2025-10-09-3494-find-the-minimum-amount-of-time-to-brew-potions/"
leetcode_ui: true
entry_slug: "2025-10-09-3494-find-the-minimum-amount-of-time-to-brew-potions"
---

[3494. Find the Minimum Amount of Time to Brew Potions](https://leetcode.com/problems/find-the-minimum-amount-of-time-to-brew-potions/) medium
[blog post](https://leetcode.com/problems/find-the-minimum-amount-of-time-to-brew-potions/solutions/7261240/kotlin-rust-by-samoylenkodmitry-js0f/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05102025-3494-find-the-minimum-amount?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/-pyLcrCIC7k)

![ba3a9efc-683f-418d-852e-1d2799bf5e36 (1).webp](/assets/leetcode_daily_images/dd1a5284.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1137

#### Problem TLDR

Min total time to finish all skills[i]*mana[j], non-intersecting #medium #bs

#### Intuition

The order is preserved.
Binary Search the start time for each potion. Check if all next times are bigger then previous.

#### Approach

* use Long.MAX_VALUE / 2 for right border of bs
* n^2 solution from lee: optimal start(mana) = max_i(finish[i+1]-mana * sum(skills[0..i]))

#### Complexity

- Time complexity:
$$O(n^2log(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 1681ms
    fun minTime(s: IntArray, m: IntArray): Long {
        val ts = LongArray(s.size + 1)
        for (p in m) {
            var lo = ts[0]; var hi = Long.MAX_VALUE / 2
            while (lo <= hi) {
                val m = (lo + hi)/2; var curr = m
                for (i in 1..<ts.size) {
                    if (curr < ts[i]) { curr = -1; break }
                    curr += s[i-1] * p
                }
                if (curr >= 0) hi = m - 1 else lo = m + 1
            }
            ts[0] = lo; for (i in 1..<ts.size) ts[i] = ts[i-1] + 1L * s[i-1] * p
        }
        return ts.last()
    }

```
```rust

// 53ms
    pub fn min_time(mut s: Vec<i32>, m: Vec<i32>) -> i64 {
        for i in 1..s.len() { s[i] += s[i - 1] }; let mut p = 0i64;
        (1..m.len()).map(|i|
            (1..s.len()).fold(s[0] as i64 * m[i - 1] as i64, |min, j|
                min.max(m[i-1] as i64 * s[j] as i64 - m[i] as i64 * s[j-1] as i64))
        ).sum::<i64>() + s[s.len()-1] as i64 * m[m.len()-1] as i64
    }

```


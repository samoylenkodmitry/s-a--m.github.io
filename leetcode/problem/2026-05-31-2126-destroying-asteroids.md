---
layout: leetcode-entry
title: "2126. Destroying Asteroids"
permalink: "/leetcode/problem/2026-05-31-2126-destroying-asteroids/"
leetcode_ui: true
entry_slug: "2026-05-31-2126-destroying-asteroids"
---

[2126. Destroying Asteroids](https://leetcode.com/problems/destroying-asteroids/solutions/8304091/kotlin-rust-by-samoylenkodmitry-drpg/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31052026-2126-destroying-asteroids?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/QTn773sIRmQ)

https://dmitrysamoylenko.com/leetcode/

![31.05.2026.webp](/assets/leetcode_daily_images/31.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1376

#### Problem TLDR

Can consume all asteroids smaller than running sum?

#### Intuition

* sort + greedy: sort, then take from smaller to larger
* bits group: consume from smallest highest one bit  group if it's min is bigger than mass
* quickselect: move to prefix & consume all asteroids that are not larger; the math guarantees O(n) - otherwise numbers have to grow by 2^i

#### Approach

* Rust's select_nth_unstable requires mid position, can't be used here

#### Complexity

- Time complexity:
$$O(nlogn|n)$$

- Space complexity:
$$O(n|1)$$

#### Code

```kotlin
    fun asteroidsDestroyed(m: Int, a: IntArray) =
    a.sorted().fold(1L*m){r,t->if(r<t)0L else r+t}>0
```
```rust
    pub fn asteroids_destroyed(m: i32, mut a: Vec<i32>) -> bool {
        let (mut c, mut s) = (m as i64, 0);
        while s < a.len() {
            let mut i = s;
            for j in s..a.len() {
                if a[j]as i64 <= c { c += a[j] as i64; a.swap(i, j); i += 1 }
            }
            if s == i { return false }; s = i
        } true
    }
```


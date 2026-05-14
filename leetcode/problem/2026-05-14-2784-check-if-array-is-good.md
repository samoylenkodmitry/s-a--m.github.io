---
layout: leetcode-entry
title: "2784. Check if Array is Good"
permalink: "/leetcode/problem/2026-05-14-2784-check-if-array-is-good/"
leetcode_ui: true
entry_slug: "2026-05-14-2784-check-if-array-is-good"
---

[2784. Check if Array is Good](https://leetcode.com/problems/check-if-array-is-good/solutions/8221832/kotlin-rust-by-samoylenkodmitry-8l8l/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14052026-2784-check-if-array-is-good?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2k6Zry9Hpbc)

https://dmitrysamoylenko.com/leetcode/

![14.05.2026.webp](/assets/leetcode_daily_images/14.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1359

#### Problem TLDR

Is combination 1..n,n

#### Intuition

* no-brain solution: check every number (1..n) is in array, and count(n)==2
* shortest solution: (1..n)+n==sorted()
* optimal solution: use array indices as visited storage

#### Approach

* n[0] can be used as extra storage for n[len-1] special case

#### Complexity

- Time complexity:
$$O(nlogn|n)$$

- Space complexity:
$$O(n|1)$$

#### Code

```kotlin
    fun isGood(n: IntArray) =
    (1..<n.size) + (n.size-1) == n.sorted()
```
```rust
    pub fn is_good(mut n: Vec<i32>) -> bool {
        (0..n.len()).all(|i| { let x = n[i].abs() as usize;
            !(x>=n.len()||n[x]<0&&(x<n.len()-1||n[0]<0)) && {
            if n[x] < 0 { n[0] *= -1 } else { n[x] *= -1 };1>0}
        }) && n[0]<0
    }
```


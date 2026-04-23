---
layout: leetcode-entry
title: "1015. Smallest Integer Divisible by K"
permalink: "/leetcode/problem/2025-11-25-1015-smallest-integer-divisible-by-k/"
leetcode_ui: true
entry_slug: "2025-11-25-1015-smallest-integer-divisible-by-k"
---

[1015. Smallest Integer Divisible by K](https://leetcode.com/problems/smallest-integer-divisible-by-k/description) medium
[blog post](https://leetcode.com/problems/smallest-integer-divisible-by-k/solutions/7373285/kotlin-rust-by-samoylenkodmitry-f4fp/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25112025-1015-smallest-integer-divisible?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Pnk2_PSD05k)

![72671a01-2ac7-463c-bbb2-e06c9dbcfc74 (1).webp](/assets/leetcode_daily_images/a10e7033.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1184

#### Problem TLDR

Smallest number 11..1 % k #medium

#### Intuition

```j
    // k 10^5
    // x*k = n, n is '1(1)`
    // return length(n) or lg(n)
    // smallest
    // 11 = 10 + 1
    // 111 / 3 = 37
    // this is math brainteaser
    //
    // what numbers gives 1 at the end
    // 1 7 11
    //
    // how to make result all of ones?
    //
    // or, if we take long string of ones how to find if it is %k
    // we have to multiply by 10 and add 1
    // can we do %k every time and check first that is 0?
    // how to stop?
```

#### Approach

* to stop look for remainder loops or just stop at k steps

#### Complexity

- Time complexity:
$$O(k)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 11ms
    fun smallestRepunitDivByK(k: Int): Int {
        var x = 0
        return (1..k).firstOrNull { x = (x * 10 + 1) % k; x == 0 } ?: -1
    }
```
```rust
// 0ms
    pub fn smallest_repunit_div_by_k(k: i32) -> i32 {
        let mut x = 0;
        (1..=k).find(|i| {x=(x*10+1)%k; x==0}).unwrap_or(-1)
    }
```


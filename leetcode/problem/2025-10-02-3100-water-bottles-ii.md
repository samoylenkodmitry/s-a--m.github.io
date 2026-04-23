---
layout: leetcode-entry
title: "3100. Water Bottles II"
permalink: "/leetcode/problem/2025-10-02-3100-water-bottles-ii/"
leetcode_ui: true
entry_slug: "2025-10-02-3100-water-bottles-ii"
---

[3100. Water Bottles II](https://leetcode.com/problems/water-bottles-ii/description) medium
[blog post](https://leetcode.com/problems/water-bottles-ii/solutions/7242154/kotlin-rust-by-samoylenkodmitry-jzlf/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02102025-3100-water-bottles-ii?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/fe7dj6FkfYM)

![1.webp](/assets/leetcode_daily_images/73968e0a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1130

#### Problem TLDR

Total drinks with growing exchange rate empty for full #medium #simulation

#### Intuition

Simulate the process. Don't forget to keep lefover empty bottles.

#### Approach

* there is a O(1) math solution exists

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 103ms
    fun maxBottlesDrunk(b: Int, x: Int, e: Int = 0): Int =
    b + if (b+e<x) 0 else maxBottlesDrunk(1,x+1,b+e-x)

```
```rust

// 3ms
    pub fn max_bottles_drunk(mut b: i32, mut x: i32) -> i32 {
        let (mut e, mut d) = (0, 0);
        while b > 0 || e >= x {
            d += b; e += b; b = if (e < x) {0} else {e-=x;x+=1;1};
        } d
    }

```


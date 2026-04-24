---
layout: leetcode-entry
title: "2833. Furthest Point From Origin"
permalink: "/leetcode/problem/2026-04-24-2833-furthest-point-from-origin/"
leetcode_ui: true
entry_slug: "2026-04-24-2833-furthest-point-from-origin"
---

[2833. Furthest Point From Origin](https://leetcode.com/problems/furthest-point-from-origin/solutions/8089767/kotlin-rust-by-samoylenkodmitry-aiic/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24042026-2833-furthest-point-from?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/MYVCEwtWLkE)

![24.04.2026.webp](/assets/leetcode_daily_images/24.04.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1338

#### Problem TLDR

Max dist when replace _ with L or R

#### Intuition

The brute-force: replace all _ to R and check balance, then to L and check again.
Some geometry transformation:_ + abs(R-L) = length-2*min(L,R) (see video explanation)

#### Approach

* asci %5 gives perfect 0,1,2 for _,L,R, do with that what you want

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun furthestDistanceFromOrigin(m: String) =
    2*"LR".maxOf{c->m.count{it!=c}}-m.length
```
```rust
    pub fn furthest_distance_from_origin(m: String) -> i32 {
        (m.len()-2*m.matches('L').count().min(m.matches('R').count())) as _
    }
```


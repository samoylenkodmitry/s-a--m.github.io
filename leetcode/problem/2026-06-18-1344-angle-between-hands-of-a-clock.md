---
layout: leetcode-entry
title: "1344. Angle Between Hands of a Clock"
permalink: "/leetcode/problem/2026-06-18-1344-angle-between-hands-of-a-clock/"
leetcode_ui: true
entry_slug: "2026-06-18-1344-angle-between-hands-of-a-clock"
---

[1344. Angle Between Hands of a Clock](https://leetcode.com/problems/angle-between-hands-of-a-clock/solutions/8341934/kotlin-rust-by-samoylenkodmitry-2u0f/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18062026-1344-angle-between-hands?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Wq31E31505Y)

https://dmitrysamoylenko.com/leetcode/

![18.06.2026.webp](/assets/leetcode_daily_images/18.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1394

#### Problem TLDR

Angle between clock arrows

#### Intuition

* 360 degrees full circle
* 360/60=6 degrees one hour
* h*60+m total minutes M
* 5.5*M%360 periodic angle between hands of a clock

#### Approach

* Rust: successors

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun angleClock(h: Int, m: Int) =
    180-abs(abs(h*30-5.5*m)-180)
```
```rust
    pub fn angle_clock(h: i32, m: i32) -> f64 {
        successors(Some(0.0f64),|a|Some((a+5.5)%360.))
        .nth((h*60+m)as usize %720).map(|a|a.min(360.-a)).unwrap()
    }
```


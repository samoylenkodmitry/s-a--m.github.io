---
layout: leetcode-entry
title: "836. Rectangle Overlap"
permalink: "/leetcode/problem/2026-09-14-836-rectangle-overlap/"
leetcode_ui: true
entry_slug: "2026-09-14-836-rectangle-overlap"
---

[836. Rectangle Overlap](https://leetcode.com/problems/rectangle-overlap/solutions/8520958/kotlin-rust-by-samoylenkodmitry-no2f/) easy
[substack](https://dmitriisamoilenko.substack.com/p/14092026-836-rectangle-overlap?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ypaM6wU5pqs)

https://dmitrysamoylenko.com/leetcode/

![14.09.2026.webp](/assets/leetcode_daily_images/14.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1482

#### Problem TLDR

Rectangle overlap?

#### Intuition

20 minutes for this easy problem.
```j
    //
    //          xxxxxxx 12,20
    //          x     x
    //          x*****x****13,15
    //          x     x     *
    //          x     x     *
    //          x     x     *
    //          xxxxxxx******
    //          7,8  10,8
    //
    //
```
Left side of intersection is max(La,Lb).
Right side of intersection is min(Ra,Rb).
Same for the Y coordinate.

#### Approach

* max(La,Lb)<min(Ra,Rb) is simplified to La < Rb && Lb < Ra

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun isRectangleOverlap(a: IntArray, b: IntArray) =
        (0..1).all{i->a[i]<b[i+2]&&b[i]<a[i+2]}
```
```rust
    pub fn is_rectangle_overlap(a: Vec<i32>, b: Vec<i32>) -> bool {
        (0..2).all(|i|a[i]<b[i+2]&&b[i]<a[i+2])
    }
```


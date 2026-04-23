---
layout: leetcode-entry
title: "812. Largest Triangle Area"
permalink: "/leetcode/problem/2025-09-27-812-largest-triangle-area/"
leetcode_ui: true
entry_slug: "2025-09-27-812-largest-triangle-area"
---

[812. Largest Triangle Area](https://leetcode.com/problems/largest-triangle-area/description/) easy
[blog post](https://leetcode.com/problems/largest-triangle-area/solutions/7227777/kotlin-rust-by-samoylenkodmitry-55cv/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27092025-812-largest-triangle-area?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/G6-rCzqV0GY)

![1.webp](/assets/leetcode_daily_images/e7e8f19c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1125

#### Problem TLDR

Max area triangle #easy

#### Intuition

Brute-force & Google for formula.

#### Approach

* max are triangle lies on a convex-hull

#### Complexity

- Time complexity:
$$O(n^2log(n))$$ or n^2

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 31ms
    fun largestTriangleArea(p: Array<IntArray>) =
        p.maxOf {(x1,y1)-> p.maxOf{(x2,y2)-> p.maxOf{(x3,y3) ->
        abs((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1)) }}} * 0.5

```

```rust

// 3ms
    pub fn largest_triangle_area(p: Vec<Vec<i32>>) -> f64 {
        let mut r = 0f64;
        for p1 in &p { for p2 in &p { for p3 in &p {
            r = r.max(((p2[0]-p1[0])*(p3[1]-p1[1])-(p3[0]-p1[0])*(p2[1]-p1[1])).abs()as f64)
        }}}; r * 0.5
    }

```


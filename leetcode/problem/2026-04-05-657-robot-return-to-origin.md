---
layout: leetcode-entry
title: "657. Robot Return to Origin"
permalink: "/leetcode/problem/2026-04-05-657-robot-return-to-origin/"
leetcode_ui: true
entry_slug: "2026-04-05-657-robot-return-to-origin"
---

[657. Robot Return to Origin](https://open.substack.com/pub/dmitriisamoilenko/p/05042026-657-robot-return-to-origin?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) easy

[youtube](https://youtu.be/LVBqcxYqViI)

![05.04.2026.webp](/assets/leetcode_daily_images/05.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1319

#### Problem TLDR

R,L,U,D return to 0 in XY plane #easy

#### Intuition

Compare counts separately for vertical and horizontal directions.
Both directions can fit into a single variable h*2^16+v.

#### Approach

* hash collisions of 8 make the solution extra spicy
* do you know how %5 works?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 38ms
    fun judgeCircle(m: String) =
        0==m.sumOf {listOf(8,-1,1,-8)[it.code%5]}
```
```rust
// 0ms
    pub fn judge_circle(m: String) -> bool {
        0==m.bytes().fold(0,|a,b|a+[8,-1,1,-8][(b%5)as usize])
    }
```


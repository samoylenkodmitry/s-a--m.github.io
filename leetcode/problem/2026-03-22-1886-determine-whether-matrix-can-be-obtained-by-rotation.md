---
layout: leetcode-entry
title: "1886. Determine Whether Matrix Can Be Obtained By Rotation"
permalink: "/leetcode/problem/2026-03-22-1886-determine-whether-matrix-can-be-obtained-by-rotation/"
leetcode_ui: true
entry_slug: "2026-03-22-1886-determine-whether-matrix-can-be-obtained-by-rotation"
---

[1886. Determine Whether Matrix Can Be Obtained By Rotation](https://open.substack.com/pub/dmitriisamoilenko/p/22032026-1886-determine-whether-matrix?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22032026-1886-determine-whether-matrix?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EVJ4x0Ji2NQ)

![33991ffb-8ced-4270-89c6-2c888059ee38 (1).webp](/assets/leetcode_daily_images/e59f5def.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1305

#### Problem TLDR

Rotate matrix #easy

#### Intuition

Rotate 4 times and check deep equals.

#### Approach

* or just check 4 counters of equal values, should be n^2

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin
// 14ms
    fun findRotation(m: Array<IntArray>, t: Array<IntArray>) =
    generateSequence(m) { m ->
        Array(m.size){ y -> IntArray(m.size) { x -> m[m.size-1-x][y] }}
    }.take(4).any(t::contentDeepEquals)
```
```rust
// 0ms
    pub fn find_rotation(m: Vec<Vec<i32>>, t: Vec<Vec<i32>>) -> bool {
        let n = m.len(); iterate(m, |c| (0..n).map(|i| (0..n).map(|j|
        c[n-1-j][i]).collect()).collect()).take(4).any(|c| c == t)
    }
```


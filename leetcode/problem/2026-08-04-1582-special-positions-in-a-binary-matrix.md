---
layout: leetcode-entry
title: "1582. Special Positions in a Binary Matrix"
permalink: "/leetcode/problem/2026-08-04-1582-special-positions-in-a-binary-matrix/"
leetcode_ui: true
entry_slug: "2026-08-04-1582-special-positions-in-a-binary-matrix"
---

[1582. Special Positions in a Binary Matrix](https://open.substack.com/pub/dmitriisamoilenko/p/04082026-1582-special-positions-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) easy
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/04082026-1582-special-positions-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04082026-1582-special-positions-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/W8c8wr4GaWw)

![7e6d1320-c887-4b63-976e-83ef9189f37b (1).webp](/assets/leetcode_daily_images/6dfe3e7e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1287

#### Problem TLDR

Count single 1-row-column #easy

#### Intuition

Brute-force.

#### Approach

* check each cell
* or check each row

#### Complexity

- Time complexity:
$$O(n^2m^2)$$, can be O(nm)

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 14ms
    fun numSpecial(m: Array<IntArray>) = m.count { r ->
        r.sum()==1 && m.sumOf { it[r.indexOf(1)] }==1
    }
```
```rust
// 0ms
    pub fn num_special(m: Vec<Vec<i32>>) -> i32 {
        m.iter().filter(|r|r.iter().sum::<i32>()==1&&
        m.iter().map(|R|R[r.iter().position(|&x|x>0).unwrap()]).sum::<i32>()==1).count() as _
    }
```


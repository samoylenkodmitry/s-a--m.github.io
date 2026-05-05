---
layout: leetcode-entry
title: "48. Rotate Image"
permalink: "/leetcode/problem/2026-05-04-48-rotate-image/"
leetcode_ui: true
entry_slug: "2026-05-04-48-rotate-image"
---

[48. Rotate Image](https://leetcode.com/problems/rotate-image/solutions/8136524/kotlin-rust-by-samoylenkodmitry-rjw1/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04052026-48-rotate-image?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/VdcTwy5zIEQ)

https://dmitrysamoylenko.com/leetcode/

![04.05.2026.webp](/assets/leetcode_daily_images/04.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1348

#### Problem TLDR

Rotate square matrix

#### Intuition

Go layer by layer and do 4-swaps in place on a single side

#### Approach

* reverse + transpose also works (in any order)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun rotate(m: Array<IntArray>): Unit {
        val n = m.size;
        for (l in 0..<n / 2) for (s in l..<n - l - 1) {
            val oi = n - 1 - l; val oj = n - 1 - s; val t = m[oi][oj]
            m[oi][oj] = m[s][oi]; m[s][oi] = m[l][s]
            m[l][s] = m[oj][l]; m[oj][l] = t
        }
    }
```
```rust
    pub fn rotate(m: &mut Vec<Vec<i32>>) {
        m.reverse();
        for i in 0..m.len() { for j in i+1..m.len() {
            let t = m[i][j]; m[i][j]=m[j][i]; m[j][i] = t
        }}
    }
```


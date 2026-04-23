---
layout: leetcode-entry
title: "1975. Maximum Matrix Sum"
permalink: "/leetcode/problem/2026-01-05-1975-maximum-matrix-sum/"
leetcode_ui: true
entry_slug: "2026-01-05-1975-maximum-matrix-sum"
---

[1975. Maximum Matrix Sum](https://leetcode.com/problems/maximum-matrix-sum/description) medium
[blog post](https://leetcode.com/problems/maximum-matrix-sum/solutions/7467857/kotlin-rust-by-samoylenkodmitry-tjzm/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05012026-1975-maximum-matrix-sum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/xHi997OssVg)

![6f7a2399-202e-420c-a2f8-a9ea9b74892d (1).webp](/assets/leetcode_daily_images/d8d6ffca.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1226

#### Problem TLDR

Max sum after multiply -1 adjacent cells #medium #brainteaser

#### Intuition

Notice that we can move "-" to any cell in the matrix.
Count "-", find sum and find minimum value to subtract.

```j
    // we can propagate - to any cell
    // if even -- count then all positive
    // otherwise make it the smallest abs
```

#### Approach

* don't forget to subtract 2*min

#### Complexity

- Time complexity:
$$O(m)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 9ms
    fun maxMatrixSum(m: Array<IntArray>): Long {
        var cnt = 0; var min = Int.MAX_VALUE
        return m.sumOf { it.sumOf { v ->
            if (v < 0) cnt++
            min = min(min, abs(v)); abs(v).toLong()
        }} - (cnt%2)*2*min
    }
```
```rust
// 0ms
    pub fn max_matrix_sum(m: Vec<Vec<i32>>) -> i64 {
        let (mut c, mut min) = (0, i64::MAX);
        m.iter().map(|r| r.iter().map(|&x|{ let v = x.abs() as i64;
            min = min.min(v); c += (x < 0) as i64; v
        }).sum::<i64>()).sum::<i64>() - 2*min*(c&1)
    }
```


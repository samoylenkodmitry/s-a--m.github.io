---
layout: leetcode-entry
title: "840. Magic Squares In Grid"
permalink: "/leetcode/problem/2025-12-30-840-magic-squares-in-grid/"
leetcode_ui: true
entry_slug: "2025-12-30-840-magic-squares-in-grid"
---

[840. Magic Squares In Grid](https://leetcode.com/problems/magic-squares-in-grid/description/) medium
[blog post](https://leetcode.com/problems/magic-squares-in-grid/solutions/7450408/kotlin-rust-by-samoylenkodmitry-t9ia/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30122025-840-magic-squares-in-grid?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Tql4KSUZEnI)

![5252578a-5336-4593-b996-35ed7a689f29 (1).webp](/assets/leetcode_daily_images/15058c46.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1220

#### Problem TLDR

Count 'magic' 3x3 squares #medium

#### Intuition

Brute-force.

#### Approach

* we can enumerate each cell and translate into original matrix m[y-1][x-1] shift by y1,x1 of the magic cell

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 24ms
    fun numMagicSquaresInside(g: Array<IntArray>) =
        (1..<g.size-1).sumOf { y -> (1..<g[0].size-1).count { x ->
            "012345678036147258048246".map { g[y-1+(it-'0')/3][x-1+(it-'0')%3] }
            .run {toSet().size==9 && all{it in 1..9} && chunked(3).all{it.sum()== 15}}
        }}
```
```rust
// 0ms
    pub fn num_magic_squares_inside(g: Vec<Vec<i32>>) -> i32 {
        (1..g.len()-1).map(|y|(1..g[0].len()-1).filter(|&x|{
            let v=[0,1,2,3,4,5,6,7,8,0,3,6,1,4,7,2,5,8,0,4,8,2,4,6]
                .map(|i| g[y-1+i/3][x-1+i%3]); let mut m=0u16;
            v[..9].iter().all(|&t|0<t&&t<=9&&{m&1<<t<1&&{m|=1<<t;true}})
            && v.chunks(3).all(|c|c.iter().sum::<i32>()==15)
        }).count()as i32).sum()
    }
```


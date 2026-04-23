---
layout: leetcode-entry
title: "1727. Largest Submatrix With Rearrangements"
permalink: "/leetcode/problem/2026-03-17-1727-largest-submatrix-with-rearrangements/"
leetcode_ui: true
entry_slug: "2026-03-17-1727-largest-submatrix-with-rearrangements"
---

[1727. Largest Submatrix With Rearrangements](https://open.substack.com/pub/dmitriisamoilenko/p/17032026-1727-largest-submatrix-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/17032026-1727-largest-submatrix-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17032026-1727-largest-submatrix-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/5CJSufeR7ts)

![13158f9f-a659-4fd4-bfdb-725ce8938b90 (1).webp](/assets/leetcode_daily_images/5f6052bc.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1300

#### Problem TLDR

Max area of binary matrix, swap columns #medium

#### Intuition

```j
    // columns!
    //         6
    // ***___***___
    // __*****_____ 5
    // ______***___
    // ______*_____
    // ______*_____
    // ______*_____
    // ______*_____
    //       7
    //
    // 001
    // 112 = 1+1+1=3
    // 203 = 2+2=4
    //
    // 22223345  max(2*8,3*4, 4*2, 5)
```

* subproblem: only the last row with histogram of the previous rows
* sort the histogram, O(n) scan max(h(x)*(width-x))

#### Approach

* don't have to build histogram for the first row
* r[x] *= prev + 1 trick

#### Complexity

- Time complexity:
$$O(nmlog(m))$$

- Space complexity:
$$O(m)$$

#### Code

```kotlin
// 110ms
    fun largestSubmatrix(m: Array<IntArray>) =
        m.withIndex().maxOf { (y,r) ->
            if (y>0) for (x in r.indices) r[x] *= m[y-1][x] + 1
            r.sorted().withIndex().maxOf { (x,v) -> (r.size-x)*v }
        }
```
```rust
// 4ms
    pub fn largest_submatrix(mut m: Vec<Vec<i32>>) -> i32 {
        (0..m.len()).map(|y| {
            if y > 0 { for x in 0..m[0].len() { m[y][x] *= m[y-1][x]+1}}
            let mut r = m[y].clone(); r.sort();
            (0..r.len()).map(|x|(r.len()-x)as i32*r[x]).max().unwrap()
        }).max().unwrap()
    }
```


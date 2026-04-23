---
layout: leetcode-entry
title: "3623. Count Number of Trapezoids I"
permalink: "/leetcode/problem/2025-12-02-3623-count-number-of-trapezoids-i/"
leetcode_ui: true
entry_slug: "2025-12-02-3623-count-number-of-trapezoids-i"
---

[3623. Count Number of Trapezoids I](https://leetcode.com/problems/count-number-of-trapezoids-i/description) medium
[blog post](https://leetcode.com/problems/count-number-of-trapezoids-i/solutions/7387060/kotlin-rust-by-samoylenkodmitry-qx8x/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02112025-3623-count-number-of-trapezoids?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/RVblRQ8lyJk)

![321670b3-31c5-41bb-bee4-2f630f5c5a78 (1).webp](/assets/leetcode_daily_images/7ce55cb4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1191

#### Problem TLDR

Trapezoids count #medium

#### Intuition

```j
    //         *              *          2
    //    *           *           *      3     2,3= 3
    //       *            *              2     2,2=1 + 2,3=3 = 4
    //             *                     1     skip
    //  *       *      *        *        4     2,4=6 + 3,4=(6+6+6)=18 + 2,4=6 = 6*5=30
    //     *       *     *       *       4     prev(30)+f(4,4)
    //        *       *      *           3     2*(3,2) + 3,3 + 2*(3,4)
    // 1 - 0
    // 2 - 1
    // 3 - 3
    // 4 - 6
    // 5 - 4+3+2+1=10 5*(5-1)/2       * * * * *
    // TLE? - yes, this is O(N^2), previous are too many
    // 35 minute, look for hints: they propose n^2 algo, there are too many groups
    //                            ugly math trick to solve this
```

#### Approach

* group by level Y
* each level count adds as c*(c-1)/2
* use previous levels sum

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 81ms
    fun countTrapezoids(p: Array<IntArray>) =
        p.groupingBy {it[1]}.eachCount().values.fold(0L to 0L){ (res,sum),c ->
        val ways = 1L*c*(c-1)/2; res + ways*sum to ways+sum }.first % 1000000007
```
```rust
// 63ms
    pub fn count_trapezoids(p: Vec<Vec<i32>>) -> i32 {
        (p.iter().map(|v| v[1]).counts().values().fold((0,0), |(r,s),&c|{
        let w = (c*(c-1)/2); (r+w*s,w+s) }).0 % 1000000007) as _
    }
```


---
layout: leetcode-entry
title: "757. Set Intersection Size At Least Two"
permalink: "/leetcode/problem/2025-11-20-757-set-intersection-size-at-least-two/"
leetcode_ui: true
entry_slug: "2025-11-20-757-set-intersection-size-at-least-two"
---

[757. Set Intersection Size At Least Two](https://leetcode.com/problems/set-intersection-size-at-least-two/description) hard
[blog post](https://leetcode.com/problems/set-intersection-size-at-least-two/solutions/7362269/kotlin-rust-by-samoylenkodmitry-y8ld/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20112025-757-set-intersection-size?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/6FtDLPwWVaQ)

![36766e29-5633-49f2-9695-b0105097bb24 (1).webp](/assets/leetcode_daily_images/70b89b6c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1179

#### Problem TLDR

Min set each interval have two values in it #hard #sorting

#### Intuition

```j
    // 1 2 3 4 5
    // * * *
    // * * * *
    //   * * * *
    //     * * *
    //       i     drop 1..3, should take two values from it +2
    //                        take the last two: 2,3
    //         i   drop 1..4, the last two is in 1..4, so skip
    //           i drop 2..5, 2 in 2..5, 3in2..5, skip
    //             drop 3..5, 2 !in 3..5, take 5, 3 in 2..5  +1

    // 1 2 3 4 5
    // * *
    //   * *
    //   * * *
    //       * *
    //     i      drop 1..2, +2 (a=1,b=2)
    //       i    drop 2..3, +1 (a=3,b=2)
    //         i  drop 2..4, skip
    //            drop 4..5, +2 a=5,b=4

    // 1 2 3 4 5 6 7 8
    // * * *
    //     * * * * *
    //         * * *
    //             * *
    //       i=2         drop 1..3, +2, a=2 b=3
    //               i=3 drop 3..7, +1, a=7 b=3 then a=3 b=7
    //                   drop 5..7, +1,
```

* sort by the ends: this tells you that interval has ended and you should do something with it

#### Approach

* no need to store visited intervals in a heap
* no need for a hashset, just use two variables
* don't even need for while loop: just check if every interval has 'a' and 'b' in it

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 25ms
    fun intersectionSizeTwo(v: Array<IntArray>): Int {
        v.sortBy { it[1] }; var a = -1; var b = -1; var res = 0
        for ((l,r) in v) {
            if (a < l) { a = if (r==b) b-1 else b; b=r; res++}
            if (a < l) { a = b-1; res++}
        }
        return res
    }
```
```rust
// 3ms
    pub fn intersection_size_two(mut v: Vec<Vec<i32>>) -> i32 {
        v.sort_by_key(|v|v[1]); let (mut a, mut b, mut r) = (-1,-1,0);
        for v in &v {
            if a < v[0] { a = if b == v[1] { b-1 } else { b }; b = v[1]; r += 1 }
            if a < v[0] { a = b - 1; r += 1 }
        }; r
    }
```


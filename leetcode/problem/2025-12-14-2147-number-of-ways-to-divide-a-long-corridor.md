---
layout: leetcode-entry
title: "2147. Number of Ways to Divide a Long Corridor"
permalink: "/leetcode/problem/2025-12-14-2147-number-of-ways-to-divide-a-long-corridor/"
leetcode_ui: true
entry_slug: "2025-12-14-2147-number-of-ways-to-divide-a-long-corridor"
---

[2147. Number of Ways to Divide a Long Corridor](https://leetcode.com/problems/number-of-ways-to-divide-a-long-corridor/description/) hard
[blog post](https://leetcode.com/problems/number-of-ways-to-divide-a-long-corridor/solutions/7413028/kotlin-rust-by-samoylenkodmitry-0wwj/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14122025-2147-number-of-ways-to-divide?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ArVh89OjsY4)

![895a38fb-0bef-4120-80b5-a786aa620ac5 (1).webp](/assets/leetcode_daily_images/1831610a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1204

#### Problem TLDR

Ways to split pairs of S #hard

#### Intuition

```j
    // p p s p p s p p
```
Chunk by pairs of 's', multiply counts of in-between 'p'.

#### Approach

* or, rearrange the problem and add the accumulated value on each new 'p'

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 34ms
    fun numberOfWays(c: String): Int {
        var a = 0; var b = 1; var s = 1
        for (c in c) if (c == 'P') b = (a * (s%2) + b) % 1000000007
            else if (++s%2>0) a = b
        return a * (s%2)
    }
```
```rust
// 3ms
    pub fn number_of_ways(c: String) -> i32 {
        let (mut a, mut b, mut even) = (0, 1, true);
        for c in c.as_bytes().chunk_by(|x, y| x == y) {
            if c[0] == b'P' {
                if even && a > 0 { b = (b + c.len()* a) % 1_000_000_007 }
            } else {
                if !even || c.len() > 1 { a = b }
                if c.len() & 1 > 0 { even = !even }
            }
        } even as i32 * a as i32
    }
```


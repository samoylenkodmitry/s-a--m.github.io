---
layout: leetcode-entry
title: "3665. Twisted Mirror Path Count"
permalink: "/leetcode/problem/2026-10-31-3665-twisted-mirror-path-count/"
leetcode_ui: true
entry_slug: "2026-10-31-3665-twisted-mirror-path-count"
---

[3665. Twisted Mirror Path Count](https://leetcode.com/problems/twisted-mirror-path-count/solutions/8595937/kotlin-rust-by-samoylenkodmitry-y35u/) medium
[substack](https://dmitriisamoilenko.substack.com/p/31102026-3665-twisted-mirror-path?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/GT3lSAGvhQY)

https://dmitrysamoylenko.com/leetcode/

![31.10.2026.webp](/assets/leetcode_daily_images/31.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1528

#### Problem TLDR

Count ways to move in grid with 1-mirrors

#### Intuition

DFS DP: cache by position and direction
Bottom up DP: keep previous row as direction down and previos value as direction right, swap on mirrors

#### Approach

* Rust allows to modify array values while iterating it

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun uniquePaths(g:Array<IntArray>): Int {
        val d = IntArray(501); d[0] = 1; var a = 0
        for (r in g) {
            var R = 0
            for (x in r.indices) {
                a = (R + d[x]) % 1000000007
                if (r[x] < 1) { d[x] = a; R = a } else R = d[x].also { d[x] = R }
            }
        }
        return a
    }
```
```rust
    pub fn unique_paths(g: Vec<Vec<i32>>) -> i32 {
        let (mut d, mut a) = ([0; 501], 0); d[0] = 1;
        for r in g {
            let mut R = 0;
            for (D, v) in d.iter_mut().zip(r) {
                a = (R + *D) % 1000000007;
                (*D, R) = if v < 1 { (a, a) } else { (R, *D) }
            }
        } a
    }
```


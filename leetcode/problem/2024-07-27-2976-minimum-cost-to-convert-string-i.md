---
layout: leetcode-entry
title: "2976. Minimum Cost to Convert String I"
permalink: "/leetcode/problem/2024-07-27-2976-minimum-cost-to-convert-string-i/"
leetcode_ui: true
entry_slug: "2024-07-27-2976-minimum-cost-to-convert-string-i"
---

[2976. Minimum Cost to Convert String I](https://leetcode.com/problems/minimum-cost-to-convert-string-i/description/) medium
[blog post](https://leetcode.com/problems/minimum-cost-to-convert-string-i/solutions/5542204/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27072024-2976-minimum-cost-to-convert?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/kXYfIcfvkQc)
![2024-07-27_09-25_1.webp](/assets/leetcode_daily_images/b20cd598.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/683

#### Problem TLDR

Min cost to change the source to the target #medium #FloydWarshall

#### Intuition

We need to find the shortest paths from char to char. The best way to find them all is the Floyd-Warshall algorithm: repeat `i` = n times to optimize the path distance: `path[j][k] = min(path[j][i] + path[i][k])`.

#### Approach

* careful with the duplicates in original to changed mapping
* we can use 127 or 26 alphabets
* Rust can't return the result from the inside of lambda

#### Complexity

- Time complexity:
$$O(n + a^3 + m)$$ where `a` is an alphabet, `m` is mapping size

- Space complexity:
$$O(a^2)$$

#### Code

```kotlin

    fun minimumCost(source: String, target: String, original: CharArray,
        changed: CharArray, cost: IntArray): Long {
        val path = Array(128) { LongArray(128) { Long.MAX_VALUE / 2 }}
        for (i in cost.indices) path[original[i].code][changed[i].code] =
            min(path[original[i].code][changed[i].code], cost[i].toLong())
        for (i in 0..127) path[i][i] = 0
        for (i in 0..127) for (j in 0..127) for (k in 0..127)
            path[j][k] = min(path[j][k], path[j][i] + path[i][k])
        return source.indices.sumOf {
            path[source[it].code][target[it].code]
            .also { if (it == Long.MAX_VALUE / 2) return -1 }}
    }

```
```rust

    pub fn minimum_cost(source: String, target: String, original: Vec<char>,
        changed: Vec<char>, cost: Vec<i32>) -> i64 {
        let (mut path, x, mut res) = (vec![vec![i64::MAX / 2; 26]; 26], 'a' as usize, 0);
        for i in 0..cost.len() {
            let a = original[i] as usize - x; let b = changed[i] as usize - x;
            path[a][b] = path[a][b].min(cost[i] as i64)
        }
        for i in 0..26 { path[i][i] = 0 }
        for i in 0..26 { for a in 0..26 { for b in 0..26 {
            path[a][b] = path[a][b].min(path[a][i] + path[i][b])
        }}}
        for (a, b) in source.chars().zip(target.chars()) {
            let (a, b) = (a as usize - x, b as usize - x); let p = path[a][b];
            if p == i64::MAX / 2 { return -1 }
            res += p
        }; res
    }

```


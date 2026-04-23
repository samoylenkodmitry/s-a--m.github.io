---
layout: leetcode-entry
title: "514. Freedom Trail"
permalink: "/leetcode/problem/2024-04-27-514-freedom-trail/"
leetcode_ui: true
entry_slug: "2024-04-27-514-freedom-trail"
---

[514. Freedom Trail](https://leetcode.com/problems/freedom-trail/description/) hard
[blog post](https://leetcode.com/problems/freedom-trail/solutions/5078209/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27042024-514-freedom-trail?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/JevbY-ivBac)
![2024-04-27_09-19.webp](/assets/leetcode_daily_images/2db90530.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/585

#### Problem TLDR

Min steps to produce `key` by rotating `ring` #hard #dynamic_programming #recursion #hash_map

#### Intuition

Let's from the current position do the full search by trying each position with give letter. The minimum path is only depending on the current position of the `ring` and position in the `key` so it can be memoized.

However, don't forget to rotate optimally, sometimes it's a left rotation:
![2024-04-27_08-36.webp](/assets/leetcode_daily_images/8be05a5a.webp)

We can store the `ring` positions ahead of time.

#### Approach

Another approach is to do a Breadth-First Search: for each `key` position store all the min-length paths and their positions. Iterate from them at the next `key` position.

#### Complexity

- Time complexity:
$$O(r^2k)$$, the worst case r^2 if all letters are the same

- Space complexity:
$$O(rk)$$

#### Code

```kotlin

    fun findRotateSteps(ring: String, key: String): Int {
        val cToPos = ring.indices.groupBy { ring[it] }
        val dp = mutableMapOf<Pair<Int, Int>, Int>()
        fun dfs(i: Int, j: Int): Int = if (j == key.length) 0 else
        dp.getOrPut(i to j) {
            1 + if (ring[i] == key[j]) dfs(i, j + 1) else {
            cToPos[key[j]]!!.minOf {
                min(abs(i - it), ring.length - abs(i - it)) + dfs(it, j + 1)
        }}}
        return dfs(0, 0)
    }

```
```rust

    pub fn find_rotate_steps(ring: String, key: String) -> i32 {
        let mut pos = vec![vec![]; 26];
        for (i, b) in ring.bytes().enumerate() { pos[(b - b'a') as usize].push(i) }
        let mut layer = vec![(0, 0)];
        for b in key.bytes() {
            let mut next = vec![];
            for &i in (&pos[(b - b'a') as usize]).iter() {
                next.push((i, layer.iter().map(|&(j, path)| {
                    let diff = if i > j { i - j } else { j - i };
                    diff.min(ring.len() - diff) + path
                }).min().unwrap()))
            }
            layer = next
        }
        (layer.iter().map(|x| x.1).min().unwrap() + key.len()) as i32
    }

```


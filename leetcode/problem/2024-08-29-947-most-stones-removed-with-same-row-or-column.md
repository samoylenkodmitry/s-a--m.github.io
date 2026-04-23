---
layout: leetcode-entry
title: "947. Most Stones Removed with Same Row or Column"
permalink: "/leetcode/problem/2024-08-29-947-most-stones-removed-with-same-row-or-column/"
leetcode_ui: true
entry_slug: "2024-08-29-947-most-stones-removed-with-same-row-or-column"
---

[947. Most Stones Removed with Same Row or Column](https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/description/) medium
[blog post](https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/solutions/5705615/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29082024-947-most-stones-removed?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/r787T_UaceQ)
![1.webp](/assets/leetcode_daily_images/74951895.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/717

#### Problem TLDR

Count islands of intersecting x and y #medium #union-find

#### Intuition

The first intuition is to build a graph of connected dots and try to explore them.

![2.png](/assets/leetcode_daily_images/dc958a82.webp)

After some meditation (or using a hint), one can see that all the connected dots are removed. Union-Find helps to find the connected islands.

#### Approach

* we can connect each with each dot in O(n^2) (Rust solution)
* or we can connect each row with each column and find how many unique rows and columns are in O(n) (Kotlin solution)

#### Complexity

- Time complexity:
$$O(n^2)$$ or $$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun removeStones(stones: Array<IntArray>): Int {
        val uf = mutableMapOf<Int, Int>()
        fun f(a: Int): Int = uf[a]?.let { if (it == a) a else
            f(it).also { uf[a] = it }} ?: a
        for ((r, c) in stones) uf[f(r)] = f(-c - 1)
        return stones.size - uf.values.map { f(it) }.toSet().size
    }

```
```rust

    pub fn remove_stones(stones: Vec<Vec<i32>>) -> i32 {
        let (mut uf, mut res) = ((0..=stones.len()).collect::<Vec<_>>(), 0);
        fn f(a: usize, uf: &mut Vec<usize>) -> usize {
            while uf[a] != uf[uf[a]] { uf[a] = uf[uf[a]] }; uf[a] }
        for i in 0..stones.len() { for j in i + 1..stones.len() {
            if stones[i][0] == stones[j][0] || stones[i][1] == stones[j][1] {
                let a = f(i, &mut uf); let b = f(j, &mut uf);
                if (a != b) { res += 1; uf[a] = b }
            }
        }}; res
    }

```


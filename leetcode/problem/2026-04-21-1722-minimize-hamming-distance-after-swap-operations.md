---
layout: leetcode-entry
title: "1722. Minimize Hamming Distance After Swap Operations"
permalink: "/leetcode/problem/2026-04-21-1722-minimize-hamming-distance-after-swap-operations/"
leetcode_ui: true
entry_slug: "2026-04-21-1722-minimize-hamming-distance-after-swap-operations"
---

[1722. Minimize Hamming Distance After Swap Operations](https://leetcode.com/problems/minimize-hamming-distance-after-swap-operations/solutions/8022636/kotlin-rust-by-samoylenkodmitry-82yt/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21042026-1722-minimize-hamming-distance?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/bUSVEeholF4)

![21.04.2026.webp](/assets/leetcode_daily_images/21.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1335

#### Problem TLDR

Diff src vs target after rearrange by rules #medium #uf

#### Intuition

```j
    // we can form connected islands of numbers
    // can change only src
    // [a,b,c,d,e,f]
    //  * *   *       island, then sort or take from hashmap-with-counter by target
    //
```
1. intersected swaps form connected islands
2. we can sort both src&target inside each island
3. we can use counting sort

#### Approach

* union-find to find islands
* path compression
* Rust: itertools `counts()`
* Kotlin: Array of hashmaps shorter than groupBy

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 68ms
    fun minimumHammingDistance(s: IntArray, t: IntArray, a: Array<IntArray>): Int {
        val u = IntArray(s.size) { it }; val fvc = Array(s.size) { HashMap<Int,Int>() }
        fun f(x: Int): Int = { if (x != u[x]) u[x] = f(u[x]); u[x] }()
        for ((a,b) in a) u[f(a)] = f(b)
        for ((i,v) in s.withIndex()) fvc[f(i)][v] = 1 + (fvc[f(i)][v] ?: 0)
        return s.indices.count { i ->
            val c = fvc[f(i)][t[i]] ?: 0
            if (c > 0) fvc[f(i)][t[i]] = c-1; c <= 0
        }
    }
```
```rust
// 33ms
    pub fn minimum_hamming_distance(s: Vec<i32>, t: Vec<i32>, a: Vec<Vec<i32>>) -> i32 {
        let mut u: Vec<_> = (0..s.len()).collect();
        fn f(u: &mut [usize], x: usize) -> usize { if u[x] != x { u[x] = f(u, u[x]) } u[x] }
        for p in a { let (x, y) = (f(&mut u, p[0] as _), f(&mut u, p[1] as _)); u[x] = y; }
        let mut m = (0..s.len()).map(|i|(f(&mut u,i),s[i])).counts();
        (0..s.len()).filter(|&i| {
            let c = m.entry((f(&mut u,i),t[i])).or_default();
            if *c > 0 { *c -= 1; false } else { true }
        }).count() as _
    }
```


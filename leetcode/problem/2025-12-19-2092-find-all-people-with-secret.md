---
layout: leetcode-entry
title: "2092. Find All People With Secret"
permalink: "/leetcode/problem/2025-12-19-2092-find-all-people-with-secret/"
leetcode_ui: true
entry_slug: "2025-12-19-2092-find-all-people-with-secret"
---

[2092. Find All People With Secret](https://leetcode.com/problems/find-all-people-with-secret/description) hard
[blog post](https://leetcode.com/problems/find-all-people-with-secret/solutions/7423867/kotlin-rust-by-samoylenkodmitry-uw26/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19122025-2092-find-all-people-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/nZ85DjcrEHM)

![5285f48e-521e-486a-a8cb-3436dfe7a340 (1).webp](/assets/leetcode_daily_images/c28b740c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1209

#### Problem TLDR

Time based secret spread in a graph #hard #uf

#### Intuition

Union-Find works, but naive gives TLE. Use path compression.

#### Approach

* to disconnect nodes collect them by time groups, then short-circuit uf[x]=x
* or collect mini-graphs by time and walk DFS from connected to 0

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 145ms
    fun findAllPeople(n: Int, m: Array<IntArray>, f: Int): List<Int> {
        m.sortBy { it[2] }; val u = HashMap<Int, Int>(); u[f] = 0
        fun f(x: Int): Int = u[x]?.let { if (it==x) x else f(it).also { u[x] = it }} ?: x
        val curr = ArrayList<Int>(); var prev = 0
        for ((x,y,t) in m) {
            if (t > prev) for (x in curr) if (f(x) != f(0)) u[x] = x
            if (t > prev) curr.clear()
            u[f(x)] = f(y); curr += x; curr += y; prev = t
        }
        return (0..<n).filter { f(it) == f(0) }
    }
```
```rust
// 20ms
    pub fn find_all_people(n: i32, mut m: Vec<Vec<i32>>, fst: i32) -> Vec<i32> {
        m.sort_unstable_by_key(|v| v[2]); let mut u: Vec<_> = (0..n as usize).collect();
        let mut c = vec![]; let mut p = 0; u[fst as usize] = 0;
        fn f(u: &mut [usize], x: usize) -> usize { if x != u[x] { u[x] = f(u, u[x])} u[x]}
        for v in m {
            let (x, y, t) = (v[0] as usize, v[1] as usize, v[2]);
            if t > p { for &x in &c { if f(&mut u, x) != f(&mut u, 0) { u[x] = x }}; c.clear() }
            let rx = f(&mut u, x); u[rx] = f(&mut u, y); c.extend([x,y]); p = t
        }
        (0..n).filter(|&i| f(&mut u, i as usize) == f(&mut u, 0)).collect()
    }
```


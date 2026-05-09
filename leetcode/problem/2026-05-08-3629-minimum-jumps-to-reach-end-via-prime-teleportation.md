---
layout: leetcode-entry
title: "3629. Minimum Jumps to Reach End via Prime Teleportation"
permalink: "/leetcode/problem/2026-05-08-3629-minimum-jumps-to-reach-end-via-prime-teleportation/"
leetcode_ui: true
entry_slug: "2026-05-08-3629-minimum-jumps-to-reach-end-via-prime-teleportation"
---

[3629. Minimum Jumps to Reach End via Prime Teleportation](https://leetcode.com/problems/minimum-jumps-to-reach-end-via-prime-teleportation/solutions/8167506/kotlin-rust-by-samoylenkodmitry-yg8b/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08052026-3629-minimum-jumps-to-reach?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/cNGDuk9eVKg)

https://dmitrysamoylenko.com/leetcode/

![08.05.2026.webp](/assets/leetcode_daily_images/08.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1353

#### Problem TLDR

Shortest path by prime factors

#### Intuition

```j
    // brainteaser
    // 10^5 total
    // sqrt(n) to check if it is prime   O(NsqrtN)
    // check every number to be % by all primes in list O(n^2)
    // run BFS
    //
    // i'll go straigth to hints, have no idea
    // hints suggesting O(n^2) solution?
    // prime *factors* for each number, not precompute primes, but factors
    // 53 minute wrong answer
    // 54 minute TLE (as expected)
    // how to prepare primes buckets? 58 minute
    // so the missing part is prime factors?
```
* prepare primes with sieve
* factorize all numbers
* prepare map from prime factor to eligible indices
* bfs using this hashmap

Another way:
* instead of preparing factors, just iterate factors inside bfs step

#### Approach

* prime sieve: outer loop i = 2..n, and mark all multipliers of i with false (not prime)
* factorization: outer loop: p = primes, and check if v % p, then add, and divide v/p until not divisible
* iterating from prime to possible values: p..max step p

#### Complexity

- Time complexity:
$$O(nlogm)$$

- Space complexity:
$$O(n+m)$$

#### Code

```kotlin
    fun minJumps(n: IntArray): Int {
        val q = ArrayDeque(setOf(0)); var res = -1; val v = hashSetOf(0); var i = 1
        val s = BooleanArray(n.max()+1) {it>1}; val m = n.indices.groupBy{n[it]}
        while (++i*i<s.size) if (s[i]) for (j in i*i..<s.size step i) s[j]=false
        while (++res>=0) for (a in 1..q.size) q.removeFirst().let { i ->
            if (i == n.size-1) return res; var p = n[i]
            for (j in setOf(i-1,i+1)) if (j in n.indices && v.add(j)) q += j
            if (s[p]&&v.add(-p)) for(j in p..<s.size step p) { m[j]?.map{if(v.add(it)) q+=it}}
        }
        return 0
    }
```
```rust
    pub fn min_jumps(n: Vec<i32>) -> i32 {
        let (mut q, mut res, mut v, mut i) = (VecDeque::from([0]), 0, HashSet::from([0]), 2);
        let mut s = vec![true;*n.iter().max().unwrap() as usize + 1]; s[1]=false;
        let m = (0..n.len()).into_group_map_by(|&i|n[i] as usize);
        while i*i < s.len() { if s[i] { for j in (i*i..s.len()).step_by(i) { s[j] = false }}; i+=1}
        loop { for a in 0..q.len() {
            let i = q.pop_front().unwrap(); if i == n.len() - 1 { return res }; let p = n[i] as usize;
            for j in [i-1, i+1] { if j < n.len() && v.insert(j) { q.push_back(j); }}
            if s[p] && v.insert(n.len()+p) {
                for &j in (p..s.len()).step_by(p).filter_map(|j|m.get(&j)).flatten() {
                    if v.insert(j) { q.push_back(j); }}}
        } res += 1 } 0
    }
```


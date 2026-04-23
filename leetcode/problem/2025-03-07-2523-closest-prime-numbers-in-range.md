---
layout: leetcode-entry
title: "2523. Closest Prime Numbers in Range"
permalink: "/leetcode/problem/2025-03-07-2523-closest-prime-numbers-in-range/"
leetcode_ui: true
entry_slug: "2025-03-07-2523-closest-prime-numbers-in-range"
---

[2523. Closest Prime Numbers in Range](https://leetcode.com/problems/closest-prime-numbers-in-range/) medium
[blog post](https://leetcode.com/problems/closest-prime-numbers-in-range/solutions/6508386/kotlin-rust-by-samoylenkodmitry-ykh2/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07032025-2523-closest-prime-numbers?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/j0i8X9eZfcQ)
![1.webp](/assets/leetcode_daily_images/161e185d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/918

#### Problem TLDR

Min diff primes pair in left..right #medium #math

#### Intuition

I didn't remember the Sieve of Eratosthenes algorithm, but brute-force with sqrt(max) optimization was accepted.
The sieve works like this:
* iterate 2..n
* skip marked as non-primes
* mark current prime with all multipliers 2..n as non-prime
* https://cp-algorithms.com/algebra/sieve-of-eratosthenes.html

#### Approach

* the naive approach has O(1) memory

#### Complexity

- Time complexity:
$$O(nlog(log(n)))$$, or O(nsqrt(n))

- Space complexity:
$$O(n)$$ or O(1)

#### Code

```kotlin

    fun closestPrimes(l: Int, r: Int) = {
        val p = IntArray(r + 1)
        for (x in 2..r) if (p[x] < 1) for (j in 2..r / x) p[x * j] = 1
        (max(2, l)..r).filter { p[it] < 1 }.windowed(2)
            .minByOrNull { it[1] - it[0] } ?: listOf(-1, -1)
    }()

```
```kotlin(O(1)memory)

    fun closestPrimes(left: Int, right: Int): IntArray {
        var p = 0; var diff = 1000000; val r = intArrayOf(-1, -1)
        for (x in left..right) {
            var i = 2; var prime = true
            while (i * i <= x && prime) if (x % i++ == 0) prime = false
            if (prime && x > 1 && p > 0 && x - p < diff) { diff = x - p; r[0] = p; r[1] = x }
            if (prime && x > 1) p = x
        }
        return r
    }

```
```rust

    pub fn closest_primes(l: i32, r: i32) -> Vec<i32> {
        let (mut p, mut g) = (vec![0; 1 + r as usize], vec![]);
        for x in 2..=r { if p[x as usize] < 1 {
            if l <= x { g.push(x); } for j in 2..=r / x { p[(x * j) as usize] = 1 }}}
        g.windows(2).min_by_key(|w| w[1] - w[0]).unwrap_or(&[-1, -1]).to_vec()
    }

```
```c++

    vector<int> closestPrimes(int l, int r) {
        int p[1000001], prev = -1, d = 1e6, a = -1, b = -1;
        for (int x = 2; x <= r; x++) if (!p[x]) {
            for (int j = 2; j * x <= r; j++) p[j * x] = 1;
            if (x < l) continue;
            if (prev != -1 && x - prev < d) d = x - prev, a = prev, b = x;
            prev = x;
        }
        return {a, b};
    }

```


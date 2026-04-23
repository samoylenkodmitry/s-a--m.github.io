---
layout: leetcode-entry
title: "2594. Minimum Time to Repair Cars"
permalink: "/leetcode/problem/2025-03-16-2594-minimum-time-to-repair-cars/"
leetcode_ui: true
entry_slug: "2025-03-16-2594-minimum-time-to-repair-cars"
---

[2594. Minimum Time to Repair Cars](https://leetcode.com/problems/minimum-time-to-repair-cars/description) medium
[blog post](https://leetcode.com/problems/minimum-time-to-repair-cars/solutions/6542375/kotlin-rust-by-samoylenkodmitry-fqg7/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16032025-2594-minimum-time-to-repair?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/cKEt0VBHL4k)
![1.webp](/assets/leetcode_daily_images/ac1cbd55.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/929

#### Problem TLDR

Min rank[i] * count^2 to fix all cars #medium #binary_search #heap

#### Intuition

Didn't solve without a hint.
My (wrong) intuition was to spread the work evenly except the lowest rank:

```j

    // DP 10^5 * 100 TLE ?
    //
    // 1 2 3 4      10
    // 1*a^2 + 2*b^2 + 3*c^2 + 4*d^2
    //   7       1       1       1  max(4*1*1, 1*7*7) 49
    //   4       2       2       2  max(4*2*2, 1*4*4) 16
    //   3       3       2       2    ? 2*3*3 > 4*2*2
    //   1       3       3       3  max(4*3*3, 1*1*1) 36
    //
    // 49 > 16 < 36     search in a parabolic space
    //
    //

    // 1 5 8      6
    // 6 0 0
    // 4 1 1
    // 2 2 2   1*2*2 5*2*2  8*2*2
    // 0 3 3

```

However, that didn't allow for the binary search as space is parabolic, and optimal is not a guarantee.

The hint: `set the max allowed time`

```j
    // hint: cnt * cnt * r = time
```

Now, we can check how many cars each rank fixes: `sqrt(r/time)`.

Another solution is a heap from `u/lee215/`:
* lets give each worker a single car to fix
* compute the time as `rank * 1 * 1` = rank
* take workers one by one over increasing time
* this gives TLE, so put workers in the buckets by ranks, now we can move time much faster `freq[rank] * rank * 1 * 1`
* do this until all cars fixed, the time is the current time

#### Approach

* computing the frequency of the ranks also speed up the binary search (Rust solution)

#### Complexity

- Time complexity:
$$O(nlog(m))$$ or O(n + 100log(m)), or O(n + cars*log(100)) for heap

- Space complexity:
$$O(1)$$, or O(100) for heap and frequency-optimized binary search

#### Code

```kotlin

    fun repairCars(ranks: IntArray, cars: Int): Long {
        var lo = 0L; var hi = Long.MAX_VALUE
        while (lo <= hi) {
            val time = lo + (hi - lo) / 2L
            val cnt = ranks.sumOf { Math.sqrt(1.0 * time / it).toLong() }
            if (cnt < cars) lo = time + 1 else hi = time - 1
        }
        return lo
    }

```
```kotlin(heap)

    fun repairCars(ranks: IntArray, cars: Int): Long {
        val q = PriorityQueue<List<Int>>(compareBy { (r, c) -> 1L * r * c * c })
        val f = IntArray(101); for (r in ranks) ++f[r]; var cnt = 0; var t = 0L
        for (r in 1..100) if (f[r] > 0) q += listOf(r, 1)
        while (cnt < cars) {
            val (r, c) = q.poll(); q += listOf(r, c + 1)
            cnt += f[r]; t = 1L * r * c * c;
        }
        return t
    }

```
```rust

    pub fn repair_cars(ranks: Vec<i32>, cars: i32) -> i64 {
        let (mut f, mut lo, mut hi, c) = (vec![0; 101], 0, i64::MAX, cars as i64);
        for r in ranks { f[r as usize] += 1; hi = hi.min(r as i64 * c * c) }
        while lo <= hi {
            let (time, mut cnt) = (lo + (hi - lo) / 2, 0);
            for r in 1..101 { cnt += f[r] * (((time / r as i64) as f64).sqrt() as i64) }
            if cnt < c { lo = time + 1 } else { hi = time - 1 }
        } lo
    }

```
```c++

    long long repairCars(vector<int>& ranks, int c) {
        long long lo = 1, hi = 100'000'000'000'000LL, f[101] = {0};
        for (int r: ranks) ++f[r], hi = min(hi, 1LL * r * c * c);
        while (lo <= hi) {
            long long t = lo + (hi - lo) / 2, cnt = 0;
            for (int r = 1; r < 101; ++r) cnt += f[r] * int(sqrt(t / r));
            cnt < c ? lo = t + 1 : hi = t - 1;
        } return lo;
    }

```


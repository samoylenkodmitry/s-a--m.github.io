---
layout: leetcode-entry
title: "2537. Count the Number of Good Subarrays"
permalink: "/leetcode/problem/2025-04-16-2537-count-the-number-of-good-subarrays/"
leetcode_ui: true
entry_slug: "2025-04-16-2537-count-the-number-of-good-subarrays"
---

[2537. Count the Number of Good Subarrays](https://leetcode.com/problems/count-the-number-of-good-subarrays/description/) medium
[blog post](https://leetcode.com/problems/count-the-number-of-good-subarrays/solutions/6655720/kotlin-rust-by-samoylenkodmitry-hpt7/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16042025-2537-count-the-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/cLzE2hFBPiE)
![1.webp](/assets/leetcode_daily_images/746658db.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/960

#### Problem TLDR

Subarrays with at least k equal pairs #medium #sliding_window #math

#### Intuition

Let's observe sliding window:

```j

    // [3,1,4,3,2,2,4], k = 2
    //                  freq
    //  i     j         3->2
    //  i         j     3->2 + 2->2          2
    //  i           j   3->2 + 2->2 + 4->2   3
    //    i         j   2->2 + 4->2          2
    //      i       j   2->2 + 4->2          2
    // when to move second pointer to shrink?
    // ****************
    //    i     j->     expand until good, + all before i count
    //     i->  j       shrink while good

```
Expand window until we get `k` equal pairs.

The hardest part is to reason about when to `shrink` the window.
The `count & shrink` technique is: we always freeze the right border and shrink left while we can. The prefix is all valid subarrays, so count all of them.

Now, how to increase and decrease the frequency?

```j

    // freq = 4, (n - 1) * (n - 2) / 2
    // freq = 5, n * (n - 1) / 2 - (n - 1) * (n - 2) / 2
    //           (n - n + 2) * (n - 1) / 2
    //                    2 * (n - 1) / 2
    //                    n - 1

```
By looking at `1 1 1 1 1` example, the pairs count is `p = f * (f - 1) / 2`. So, the diff is `p(n) - p(n - 1) = n - 1`.

#### Approach

* we can shrink window up to the `invalid` state and `not` check if it is valid to add `j`, as `j = 0` initially

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun countGood(n: IntArray, k: Int): Long {
        val f = HashMap<Int, Int>(); var j = 0; var p = 0
        return n.indices.sumOf { i ->
            p += f[n[i]] ?: 0
            f[n[i]] = 1 + (f[n[i]] ?: 0)
            while (p >= k) { f[n[j]] = f[n[j]]!! - 1; p -= f[n[j++]]!! }
            1L * j
        }
    }

```
```rust

    pub fn count_good(n: Vec<i32>, mut k: i32) -> i64 {
        let (mut f, mut j) = (HashMap::new(), 0);
        (0..n.len()).map(|i| {
            k -= f.get(&n[i]).unwrap_or(&0); *f.entry(n[i]).or_default() += 1;
            while k <= 0 { *f.entry(n[j]).or_default() -= 1; k += f[&n[j]]; j += 1 }
            j as i64
        }).sum::<i64>()
    }

```
```c++

    long long countGood(vector<int>& n, int k) {
        long long r = 0LL; unordered_map<int, int> f;
        for (int i = 0, j = 0; i < size(n); ++i, r += j) {
            k -= f[n[i]]++; while (k <= 0) k += --f[n[j++]];
        } return r;
    }

```


---
layout: leetcode-entry
title: "1497. Check If Array Pairs Are Divisible by k"
permalink: "/leetcode/problem/2024-10-01-1497-check-if-array-pairs-are-divisible-by-k/"
leetcode_ui: true
entry_slug: "2024-10-01-1497-check-if-array-pairs-are-divisible-by-k"
---

[1497. Check If Array Pairs Are Divisible by k](https://leetcode.com/problems/check-if-array-pairs-are-divisible-by-k/description/) medium
[blog post](https://leetcode.com/problems/check-if-array-pairs-are-divisible-by-k/solutions/5855064/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01102024-1497-check-if-array-pairs?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/V416V8R_bKg)
![1.webp](/assets/leetcode_daily_images/8b5ce40b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/753

#### Problem TLDR

Can all pairs sums be `k`-even? #medium #modulo

#### Intuition

Modulo operation is associative, so `(a + b) % k == a % k + b % k`, the task is to find a pair for each number `x % k`: `(k - x % k) % k`.

```j

    // -4 -7 5 2 9 1 10 4 -8 -3  k=3
    //  *          *           -4=-1=2 : [1]
    //     *         *         -7=-1=2 : [1]
    //       *          *       5=2:[1]
    //         *           *    2=2:[1]
    //           *            * 9=0:[0]
    // -1 -1 2 2 0 1 1  1 -2  0 x % k
    //  2  2 2 2 0 1 1  1  1  0 (k + x % k) % k

```

The corner case is `0`, add extra `% k` to the expected value.

#### Approach

* try to solve it by hands first to feel the intuition

#### Complexity

- Time complexity:
$$O(n + k)$$

- Space complexity:
$$O(k)$$

#### Code

```kotlin

    fun canArrange(arr: IntArray, k: Int): Boolean {
        val expected = IntArray(k); var count = 0
        for (x in arr) {
            val e = (k + x % k) % k
            if (expected[e] > 0) {
                count++ ; expected[e]--
            } else expected[(k - e) % k]++
        }
        return count == arr.size / 2
    }

```
```rust

    pub fn can_arrange(arr: Vec<i32>, k: i32) -> bool {
        let (mut exp, mut cnt) = (vec![0; k as usize], 0);
        for x in &arr {
            let e = ((k + x % k) % k) as usize;
            if exp[e] > 0 { cnt += 1; exp[e] -= 1 }
            else { exp[((k - e as i32) % k) as usize] += 1 }
        }
        cnt == arr.len() / 2
    }

```
```c++

    bool canArrange(vector<int>& arr, int k) {
        vector<int> exp(k); int cnt = 0;
        for (const auto x : arr) {
            int e = (k + x % k) % k;
            if (exp[e] > 0) { cnt++; exp[e]--; }
            else exp[(k - e) % k]++;
        }
        return cnt == arr.size() / 2;
    }

```


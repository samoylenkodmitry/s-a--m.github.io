---
layout: leetcode-entry
title: "3201. Find the Maximum Length of Valid Subsequence I"
permalink: "/leetcode/problem/2025-07-16-3201-find-the-maximum-length-of-valid-subsequence-i/"
leetcode_ui: true
entry_slug: "2025-07-16-3201-find-the-maximum-length-of-valid-subsequence-i"
---

[3201. Find the Maximum Length of Valid Subsequence I](https://leetcode.com/problems/find-the-maximum-length-of-valid-subsequence-i/description/) medium
[blog post](https://leetcode.com/problems/find-the-maximum-length-of-valid-subsequence-i/solutions/6965373/kotlin-rust-by-samoylenkodmitry-4u79/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16072025-3201-find-the-maximum-length?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/xDn-X9CWQwk)
![1.webp](/assets/leetcode_daily_images/2c3aff0e.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1051

#### Problem TLDR

Longest same-pair-parity subsequence #medium #greedy

#### Intuition

4 cases to consider, take greedily
```j
    // 00000   -- valid (0+0=0)
    // 1111111 -- valid (1+1=0)
    // 010101  -- valid (1+0=1)
    // 101010 -- valid (1+0=1)
    // 0001010 -- invalid
    // 0001111 -- invalid
```

#### Approach

* write CPU-branchless with bit tricks
* use a single array for all 4 cases
* two alterating cases 0-1 and 1-0 collapses into single with initial condition of first element

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 14ms
    fun maximumLength(n: IntArray): Int {
        val c = IntArray(4)
        for (x in n) { ++c[x % 2]; c[2 + x % 2] = 1 + c[3 - x % 2] }
        return c.max()
    }

```
```kotlin

// 8ms
    fun maximumLength(n: IntArray): Int {
        var zo = 0; var oz = 0
        var zonext = 0; var oznext = 1
        var allZeros = 0; var allOnes = 0
        for (x in n) {
            allZeros += 1 - (x % 2)
            allOnes += x % 2
            if (x % 2 == zonext) { ++zo; zonext = 1 - zonext }
            if (x % 2 == oznext) { ++oz; oznext = 1 - oznext }
        }
        return maxOf(allZeros, allOnes, zo, oz)
    }

```
```kotlin

// 2ms
    fun maximumLength(n: IntArray): Int {
        var az = 0; var ao = 0; var c = 0; var e = n[0] and 1
        for (x in n) {
            val p = x and 1; az += 1 - p; ao += p
            c += 1 - p xor e; e = e xor (1 - p xor e)
        }
        return max(max(az, ao), c)
    }

```
```rust

// 0ms
    pub fn maximum_length(n: Vec<i32>) -> i32 {
        let (mut a, mut b, mut c, mut e) = (0, 0, 0, n[0] & 1);
        for x in n {
            let p = x & 1; a += 1 - p; b += p;
            c += 1 - p ^ e; e = e ^ (1 - p ^ e);
        } a.max(b).max(c)
    }

```
```c++

// 0ms
    int maximumLength(vector<int>& n) {
        int a = 0, b = 0, c = 0, e = n[0] & 1;
        for (int x: n) a += 1 - x&1, b += x&1, c += (x&1) == e, e = e ^ ((x&1) == e);
        return max(max(a, b), c);
    }

```


---
layout: leetcode-entry
title: "2094. Finding 3-Digit Even Numbers"
permalink: "/leetcode/problem/2025-05-12-2094-finding-3-digit-even-numbers/"
leetcode_ui: true
entry_slug: "2025-05-12-2094-finding-3-digit-even-numbers"
---

[2094. Finding 3-Digit Even Numbers](https://leetcode.com/problems/finding-3-digit-even-numbers/description/) easy
[blog post](https://leetcode.com/problems/finding-3-digit-even-numbers/solutions/6736561/kotlin-rust-by-samoylenkodmitry-8ayk/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12052025-2094-finding-3-digit-even?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Y7g9T3TzluI)
![1.webp](/assets/leetcode_daily_images/e82bfd82.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/986

#### Problem TLDR

Triple digit even numbers from digits #easy #backtrack

#### Intuition

This problem is not easy if you didn't spot the problem size of `1000` possible number total.

The backtracking works and is the fastest: pick a digit one-by-one, increase counter, compare with frequency, decrease back after.

#### Approach

* 3-loop solution is also possible (but 2ms vs 1ms of backtracking in Kotlin, which is unexplainable to me rn)

#### Complexity

- Time complexity:
$$O(range)$$ or O(9^3) for backtracking DFS: depth is 3, 9 digits each

- Space complexity:
$$O(answer)$$

#### Code

```kotlin

// 41ms
    fun findEvenNumbers(digits: IntArray) =
        (100..999 step 2).filter { n ->
            "$n".groupBy { it }.all { (k, v) -> v.size <= digits.count { it == k - '0' }}
        }

```
```kotlin

// 2ms
    fun findEvenNumbers(digits: IntArray): IntArray {
        val f = IntArray(10); for (d in digits) ++f[d]
        val r = IntArray(450); var i = 0
        for (a in 1..9) if (f[a] > 0) {
            f[a]--
            for (b in 0..9) if (f[b] > 0) {
                f[b]--
                for (c in 0..9 step 2) if (f[c] > 0) r[i++] = a * 100 + b * 10 + c
                f[b]++
            }
            f[a]++
        }
        return r.copyOf(i)
    }

```
```kotlin

// 1ms https://leetcode.com/problems/finding-3-digit-even-numbers/submissions/1631643083
    fun findEvenNumbers(digits: IntArray): List<Int> {
        val f = IntArray(10); for (d in digits) ++f[d]
        val res = ArrayList<Int>(); val taken = IntArray(10)
        fun dfs(soFar: Int, start: Int) {
            if (soFar > 99) {
                if (soFar % 2 == 0) res += soFar
            } else for (d in start..9) if (taken[d] < f[d]) {
                taken[d]++; dfs(soFar * 10 + d, 0); taken[d]--
            }
        }
        dfs(0, 1)
        return res
    }

```
```rust

// 0ms
    pub fn find_even_numbers(digits: Vec<i32>) -> Vec<i32> {
        let (mut f, mut r) = ([0; 10], vec![]);
        for d in digits { f[d as usize] +=  1}
        for a in 1..10 { if f[a] > 0 { f[a] -= 1;
            for b in 0..10 { if f[b] > 0 { f[b] -= 1;
                for c in (0..10).step_by(2) { if f[c] > 0 {
                    r.push((a * 100 + b * 10 + c) as i32 )}}
             f[b] += 1 }}
        f[a] += 1 }}; r
    }

```
```c++

// 0ms
    vector<int> findEvenNumbers(vector<int>& digits) {
        int f[10]={}; vector<int> r; for (int& d: digits) ++f[d];
        for (int x = 100; x < 1000; x += 2) {
            int c[10]={}, g = 1, d = x; while (d) g &= ++c[d % 10] <= f[d % 10], d /= 10;
            if (g) r.push_back(x);
        }; return r;
    }

```


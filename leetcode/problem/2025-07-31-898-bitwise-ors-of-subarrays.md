---
layout: leetcode-entry
title: "898. Bitwise ORs of Subarrays"
permalink: "/leetcode/problem/2025-07-31-898-bitwise-ors-of-subarrays/"
leetcode_ui: true
entry_slug: "2025-07-31-898-bitwise-ors-of-subarrays"
---

[898. Bitwise ORs of Subarrays](https://leetcode.com/problems/bitwise-ors-of-subarrays/description) medium
[blog post](https://leetcode.com/problems/bitwise-ors-of-subarrays/solutions/7027650/kotlin-rust-by-samoylenkodmitry-pehv/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31072025-898-bitwise-ors-of-subarrays?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/jNg4yiXboEE)
![1.webp](/assets/leetcode_daily_images/f04f9f72.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1066

#### Problem TLDR

Uniq OR-subarrays #medium #dp #bits

#### Intuition

```j
    // 101
    // 111
    // 010
    //
    // 001
    // 010
    // 110
    // 100    brainteaser, what's the rule?

    // 1000
    // 1010
    // 1001
    // 1101  should have some bits that are not filled previosly
    //       and propagate this bits up to the latest index where this bit was seen

```
My idea:
* store last visited bits positions
* for each new bit propagate it up to last visited

Other ideas:
* brute-force, but keep intermediate results in a set to reduce space and time
* same as mine core, but instead of tracking bits, modify array and go up until a[j] | a[i] != a[j]

#### Approach

* clever optimization from https://leetcode.com/problems/bitwise-ors-of-subarrays/solutions/166832/c-simplest-fastest-224-ms/ by using the fact: each new bit increases the number, so numbers are growing, no need for set (but still need at the end, as it is a series of growing parts)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(r)$$

#### Code

```kotlin

// 165ms
    fun subarrayBitwiseORs(n: IntArray): Int {
        val s = ArrayList<Int>(); var a = 0; var b = 0
        for (x in n) {
            a = b; b = s.size; s += x
            for (j in a..<b) if (s.last() != s[j] or x) s += s[j] or x
        }
        return s.toSet().size
    }

```
```kotlin

// 155ms
    fun subarrayBitwiseORs(a: IntArray): Int {
        val s = a.toHashSet(); val l = IntArray(30)
        for ((i, x) in a.withIndex()) {
            var j = i; var c = x
            for (b in 0..29) if (x shr b and 1 > 0) {
                while (j > l[b]) { c = c or a[--j]; s += c }
                l[b] = i
            }
        }
        return s.size
    }

```
```kotlin

// 126ms
    fun subarrayBitwiseORs(a: IntArray): Int {
        val s = a.toHashSet()
        for (i in a.indices) {
            var j = i - 1
            while (j >= 0 && a[i] or a[j] != a[j]) { a[j] = a[i] or a[j]; s += a[j--] }
        }
        return s.size
    }

```
```rust

// 58ms
    pub fn subarray_bitwise_o_rs(a: Vec<i32>) -> i32 {
        let (mut s, mut l) = (vec![], [0; 30]);
        for i in 0..a.len() {
            let x = a[i]; let (mut j, mut c) = (i, x); s.push(x);
            for b in 0..30 { if x >> b & 1 > 0 {
                while j > l[b] { j -= 1; c |= a[j]; s.push(c); }
                l[b] = i
            }}
        } s.sort_unstable(); s.dedup(); s.len() as _
    }

```
```rust

// 42ms
    pub fn subarray_bitwise_o_rs(mut a: Vec<i32>) -> i32 {
        let mut s = vec![];
        for i in 0..a.len() {
            let mut j = i - 1; s.push(a[i]);
            while j < a.len() && a[i] | a[j] != a[j] { a[j] |= a[i]; s.push(a[j]); j -= 1 }
        } s.sort_unstable(); s.dedup(); s.len() as _
    }

```
```c++

// 256ms
    int subarrayBitwiseORs(vector<int>& n) {
        unordered_set<int> s;
        for (int i = 0; i < size(n); ++i) {
            s.insert(n[i]);
            for (int j = i - 1; j >= 0 && ((n[i] | n[j]) != n[j]); --j)
                n[j] |= n[i], s.insert(n[j]);
        } return size(s);
    }

```
```python3

// 499ms
    def subarrayBitwiseORs(self, a: List[int]) -> int:
        s,o=set(),set();[s.update(o:={x|y for y in o}|{x}) for x in a]; return len(s)

```

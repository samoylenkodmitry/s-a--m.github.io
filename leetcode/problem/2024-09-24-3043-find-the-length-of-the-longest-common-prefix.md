---
layout: leetcode-entry
title: "3043. Find the Length of the Longest Common Prefix"
permalink: "/leetcode/problem/2024-09-24-3043-find-the-length-of-the-longest-common-prefix/"
leetcode_ui: true
entry_slug: "2024-09-24-3043-find-the-length-of-the-longest-common-prefix"
---

[3043. Find the Length of the Longest Common Prefix](https://leetcode.com/problems/find-the-length-of-the-longest-common-prefix/description/) medium
[blog post](https://leetcode.com/problems/find-the-length-of-the-longest-common-prefix/solutions/5827088/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24092024-3043-find-the-length-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/FzLnzxGHP8U)
![1.webp](/assets/leetcode_daily_images/f52b0415.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/745

#### Problem TLDR

Common digit prefix between all pairs of two num arrays #medium #trie

#### Intuition

We can construct a Trie with every suffix from arr1 and check every suffix from arr2.
Another, more short solution is to use a HashSet, and add all prefixes.

#### Approach

* we should add and check `every` prefix
* small optimization is to stop adding prefixes to HashSet as soon as current already here

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun longestCommonPrefix(arr1: IntArray, arr2: IntArray): Int {
        val set = HashSet<Int>()
        for (n in arr1) { var x = n; while (x > 0 && set.add(x)) x /= 10 }
        return arr2.maxOf { n ->
            var x = n; var pow = -1; var i = 0
            while (x > 0) {
                if (pow < 0) if (x in set) pow = i
                x /= 10; i++
            }
            if (pow < 0) 0 else i - pow
        }
    }

```
```rust

    pub fn longest_common_prefix(arr1: Vec<i32>, arr2: Vec<i32>) -> i32 {
        #[derive(Clone, Default)] struct Trie([Option<Box<Trie>>; 10]);
        let mut root: Trie = Default::default();
        let dig = |n| { let (mut x, mut dig) = (n, vec![]);
            while x > 0 { dig.push((x % 10) as usize); x /= 10 }; dig };
        for n in arr1 {
            let mut t = &mut root;
            for &d in dig(n).iter().rev() {
                t = t.0[d].get_or_insert_with(|| Box::new(Default::default()))
            }
        }
        arr2.into_iter().map(|n| {
            let (mut t, mut i) = (&root, 0);
            for &d in dig(n).iter().rev() {
                let Some(next) = t.0[d].as_ref() else { break };
                t = &next; i += 1
            }; i
        }).max().unwrap_or(0) as i32
    }

```
```c++

    int longestCommonPrefix(vector<int>& arr1, vector<int>& arr2) {
        unordered_set<int> set; int res = 0;
        for (auto& x: arr1) while (x > 0 && set.insert(x).second) x /= 10;
        for (auto& x: arr2) {
            int pow = -1; int i = 0;
            while (x > 0) {
                if (pow < 0 && set.find(x) != set.end()) pow = i;
                x /= 10; i++;
            }
            if (pow >= 0) res = max(res, i - pow);
        }
        return res;
    }

```


---
layout: leetcode-entry
title: "2349. Design a Number Container System"
permalink: "/leetcode/problem/2025-02-08-2349-design-a-number-container-system/"
leetcode_ui: true
entry_slug: "2025-02-08-2349-design-a-number-container-system"
---

[2349. Design a Number Container System](https://leetcode.com/problems/design-a-number-container-system/description/) medium
[blog post](https://leetcode.com/problems/design-a-number-container-system/solutions/6392764/kotlin-rust-by-samoylenkodmitry-4raz/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08022025-2349-design-a-number-container?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/JtUkMNCdhBE)
![1.webp](/assets/leetcode_daily_images/55c1e002.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/889

#### Problem TLDR

Smallest running index of number in map #medium #treeset

#### Intuition

To keep all indices for the number in a sorted order use a TreeSet. Store (index, number) map to remove the old number from index.

#### Approach

* one small optimization is to remove old number lazily: keep removing if m[find(n)] != n

#### Complexity

- Time complexity:
$$O(nlog(n))$$ for all operation, log(n) for `change`, O(1) for find, reverse for lazy.

- Space complexity:
$$O(n)$$ indices & numbers are never erased

#### Code

```kotlin

class NumberContainers() {
    val iin = HashMap<Int, Int>()
    val nii = HashMap<Int, TreeSet<Int>>()
    fun change(i: Int, n: Int) {
        iin[i]?.let { nii[it]!! -= i }; iin[i] = n
        nii.getOrPut(n, ::TreeSet) += i
    }
    fun find(n: Int) = nii[n]?.firstOrNull() ?: -1
}

```
```rust

#[derive(Default)] struct NumberContainers(HashMap<i32, i32>, HashMap<i32, BTreeSet<i32>>);
impl NumberContainers {
    fn new() -> Self { Self::default() }
    fn change(&mut self, i: i32, n: i32) {
        self.0.insert(i, n).inspect(|j| { self.1.get_mut(&j).unwrap().remove(&i);});
        self.1.entry(n).or_default().insert(i);
    }
    fn find(&self, n: i32) -> i32
        { *self.1.get(&n).and_then(|s| s.first()).unwrap_or(&-1) }
}

```
```c++

class NumberContainers {
    unordered_map<int, int> in; map<int, set<int>> ni;
public:
    void change(int i, int n) {
        if (in.count(i)) ni[in[i]].erase(i);
        in[i] = n, ni[n].insert(i);
    }
    int find(int n) { return size(ni[n]) ? *begin(ni[n]) : -1; }
};

```


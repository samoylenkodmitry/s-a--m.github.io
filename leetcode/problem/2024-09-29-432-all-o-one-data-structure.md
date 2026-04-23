---
layout: leetcode-entry
title: "432. All O`one Data Structure"
permalink: "/leetcode/problem/2024-09-29-432-all-o-one-data-structure/"
leetcode_ui: true
entry_slug: "2024-09-29-432-all-o-one-data-structure"
---

[432. All O`one Data Structure](https://leetcode.com/problems/all-oone-data-structure/description/) hard
[blog post](https://leetcode.com/problems/all-oone-data-structure/solutions/5848184/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29092024-432-all-oone-data-structure?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/MCiKcc6NHn4)
![1.webp](/assets/leetcode_daily_images/354748c7.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/750

#### Problem TLDR

Count usage frequencies in O(1) #hard #hashmap #linked_list

#### Intuition

The logN solution is to put buckets in a TreeMap with the keys of frequencies.

The O(1) solution is to use a doubly linked list for the buckets: it works because we only doing `inc` and `dec` operations, so at most shift by one position happens.

#### Approach

This is all about the implementation details.
* logN solution is shorter
* I've implemented O(n) solution only in Kotlin and it took me more than 6 hours to make it working and concise
* start with writing the `inc` method, after it works, write the `dec`; only after that try to extract the common logic

#### Complexity

- Time complexity:
$$O(n)$$ for n calls

- Space complexity:
$$O(n)$$

#### Code

```kotlin

class AllOne(): TreeMap<Int, HashSet<String>>() {
    val keyToFreq = HashMap<String, Int>()
    fun update(key: String, inc: Int) {
        val currFreq = keyToFreq.remove(key) ?: 0
        get(currFreq)?.let { it.remove(key); if (it.isEmpty()) remove(currFreq) }
        val newFreq = currFreq + inc
        if (newFreq > 0) getOrPut(newFreq) { HashSet() } += key
        if (newFreq > 0) keyToFreq[key] = newFreq
    }
    fun inc(key: String) = update(key, 1)
    fun dec(key: String) = update(key, -1)
    fun getMaxKey() = if (isEmpty()) "" else lastEntry().value.first()
    fun getMinKey() = if (isEmpty()) "" else firstEntry().value.first()
}

```
```kotlin

class AllOne() {
    class Node(val f: Int, var l: Node? = null, var r: Node? = null): HashSet<String>()
    operator fun Node.set(i: Int, n: Node?) = if (i < 1) l = n else r = n
    operator fun Node.get(i: Int) = if (i < 1) l else r
    val keyToNode = HashMap<String, Node?>(); var max = Node(0); var min = max;
    fun inc(key: String) {
        val curr = keyToNode[key] ?: if (min.f > 0) Node(0, r = min).also { min = it } else min
        val next = getOrInsertNext(curr, 1)
        update(curr, next, key)
        if (curr === max) max = next
    }
    fun dec(key: String) {
        var curr = keyToNode[key] ?: return
        val next = if (curr.f == 1) null else getOrInsertNext(curr, -1)
        update(curr, next, key)
    }
    fun getOrInsertNext(curr: Node, inc: Int, r: Int = (inc + 1) / 2) =
        curr[r]?.takeIf { it.f == curr.f + inc } ?: Node(curr.f + inc).apply {
            this[1 - r] = curr; this[r] = curr[r]
            curr[r] = this; this[r]?.set(1 - r, this)
        }
    fun update(curr: Node, next: Node?, key: String) {
        curr -= key; next?.add(key); keyToNode[key] = next
        if (curr.size > 0) return
        curr.l?.r = curr.r.also { curr.r?.l = curr.l }
        if (curr === max) max = next ?: curr.r ?: Node(0)
        if (curr === min) min = next ?: curr.r ?: Node(0)
    }
    fun getMaxKey() = max.firstOrNull() ?: ""
    fun getMinKey() = min.firstOrNull() ?: ""
}

```
```rust

#[derive(Default)]
struct AllOne(BTreeMap<i32, HashSet<String>>, HashMap<String, i32>);
impl AllOne {
    fn new() -> Self { Self::default() }
    fn update(&mut self, key: String, inc: i32) {
        let curr_freq = self.1.remove(&key).unwrap_or(0);
        if let Some(set) = self.0.get_mut(&curr_freq) {
            set.remove(&key);
            if set.is_empty() { self.0.remove(&curr_freq); }
        }
        let new_freq = curr_freq + inc;
        if new_freq > 0 {
            self.0.entry(new_freq).or_insert_with(HashSet::new).insert(key.clone());
            self.1.insert(key, new_freq);
        }
    }
    fn inc(&mut self, key: String) { self.update(key, 1) }
    fn dec(&mut self, key: String) { self.update(key, -1) }
    fn get_max_key(&self) -> String { self.0.iter().next_back()
      .and_then(|(_, set)| set.iter().next()).cloned().unwrap_or_default() }
    fn get_min_key(&self) -> String { self.0.iter().next()
      .and_then(|(_, set)| set.iter().next()).cloned().unwrap_or_default() }
}

```
```c++

class AllOne {
public:
    map<int, unordered_set<string>> tree;
    unordered_map<string, int> keyToFreq;
    void update(const string& key, int inc) {
        auto it = keyToFreq.find(key);
        int currFreq = (it != keyToFreq.end()) ? it->second : 0;
        keyToFreq.erase(key);
        auto& set = tree[currFreq]; set.erase(key);
        if (set.empty()) tree.erase(currFreq);
        int newFreq = currFreq + inc;
        if (newFreq > 0) { tree[newFreq].insert(key); keyToFreq[key] = newFreq; }
    }
    void inc(const string& key) { update(key, 1); }
    void dec(const string& key) { update(key, -1); }
    string getMaxKey() { return tree.empty() ? "" : *tree.rbegin()->second.begin(); }
    string getMinKey() { return tree.empty() ? "" : *tree.begin()->second.begin(); }
};

```


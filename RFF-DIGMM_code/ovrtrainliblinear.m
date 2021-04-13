function [model] = ovrtrainliblinear(y, x, cmd)

labelSet = unique(y);
labelSetSize = length(labelSet);
models = cell(labelSetSize,1);

for i=1:labelSetSize
    models{i} = train(double(y == labelSet(i)), sparse(x), cmd);
end

model = struct('models', {models}, 'labelSet', labelSet);
